import numpy as np
import pandas as pd

pd.options.display.precision = 15
import warnings

warnings.filterwarnings("ignore")
import warnings

warnings.simplefilter('ignore')
import time
from numba import jit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import gc
from .third_party import fast_auc, eval_auc, plot_importance
from sklearn.utils import shuffle


def train_model_classification_vb(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None,
                                  plot_feature_importance=False, model=None,
                                  verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None,
                                  averaging='usual', n_jobs=-1, groups=None,
                                  train_1_sample_coef=None, train_0_sample_coef=None,
                                  categorial_columns=None, categorial_encoder=None
                                  ):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to   plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else splits
    if X_test is not None:
        X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }

    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        if X_test is not None:
            prediction = np.zeros((len(X_test), 1))

    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        if X_test is not None:
            prediction = np.zeros((len(X_test), 1))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    if groups is None:
        splitter = folds.split(X, y)
    else:
        splitter = folds.split(X, y, groups=groups)
    for fold_n, (train_index, valid_index) in enumerate(splitter):
        gc.collect()
        if fold_n < folds.n_splits - n_splits:
            continue
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # encode categorial features
        if categorial_encoder is not None:
            print('Encoding categorical features')
            assert categorial_columns is not None
            encoder = categorial_encoder(verbose=1, cols=categorial_columns)
            encoder.fit(X_train, y_train)
            X_train = encoder.transform(X_train)
            X_valid = encoder.transform(X_valid)
            X_test = encoder.transform(X_test)

        # down/upsample train dataset
        if train_0_sample_coef is not None or train_1_sample_coef is not None:
            assert NotImplementedError
            if type(X_train) is np.ndarray:
                assert NotImplementedError
                # train = np.hstack([X_train.values, y_train.reshape((-1,1))])
            else:
                X_train['target'] = y_train

            train_0 = X_train[X_train.target == 0]
            train_1 = X_train[X_train.target == 1]
            if train_0_sample_coef is not None:
                train_0 = train_0.sample(int(train_0.shape[0] * train_0_sample_coef),
                                         random_state=params['random_state'], replace=True)
            if train_1_sample_coef is not None:
                train_1 = train_1.sample(int(train_1.shape[0] * train_1_sample_coef),
                                         random_state=params['random_state'], replace=True)
            train = pd.concat([train_0, train_1], axis=0)
            train = shuffle(train, random_state=params['random_state'])
            X_train = train.drop(['target'], axis=1)
            y_train = train.target
            del train_0
            del train_1
            del train

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            if X_test is not None:
                y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

        if model_type == 'xgb':
            if columns is None:
                feature_names = X.columns
            else:
                feature_names = columns
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=feature_names)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=feature_names),
                                         ntree_limit=model.best_ntree_limit)
            if X_test is not None:
                y_pred = model.predict(xgb.DMatrix(X_test, feature_names=feature_names),
                                       ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)[:, 1].reshape(-1, )
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            if X_test is not None:
                y_pred = model.predict_proba(X_test)[:, 1]

        if model_type == 'keras':
            from keras.models import Model
            from keras.callbacks import Callback, EarlyStopping

            kmodel: Model = model()

            my_callbacks = [EarlyStopping(monitor='aucroc', patience=early_stopping_rounds, verbose=1, mode='max')]

            keras_params = {k: v for k, v in params.items() if k not in ('random_state',)}
            kmodel.fit(X_train, y_train, validation_data=(X_valid, y_valid), **keras_params, callbacks=my_callbacks)
            predict_params = {k: v for k, v in params.items() if k in ['batch_size', 'verbose', 'steps', 'callbacks',
                                                                       'max_queue_size', 'workers',
                                                                       'use_multiprocessing']}
            y_pred_valid = kmodel.predict(X_valid) #, **predict_params
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            if X_test is not None:
                y_pred = kmodel.predict(X_test)[:,0]

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=categorial_columns,
                      use_best_model=True,
                      verbose=True)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            if X_test is not None:
                y_pred = model.predict_proba(X_test)[:, 1]

        if averaging == 'usual':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            if X_test is not None:
                prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            if X_test is not None:
                prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    if X_test is not None:
        prediction /= n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    if X_test is not None:
        result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
            plot_importance(result_dict)

    return result_dict
