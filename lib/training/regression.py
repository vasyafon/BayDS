def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='l2', columns=None,
                           plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None,
                           averaging='usual', n_jobs=-1, groups=None):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
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
        spliter = folds.split(X, y, groups=groups)
    for fold_n, (train_index, valid_index) in enumerate(spliter):
        if fold_n >= n_splits:
            break
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

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
            raise Exception("Implement xgb Regression!")
            # train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            # valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)
            #
            # watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            # model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
            #                   early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            # y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
            #                              ntree_limit=model.best_ntree_limit)
            # if X_test is not None:
            #     y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            raise Exception("Implement sklearn Regression!")
        #     model = model
        #     model.fit(X_train, y_train)
        #
        #     y_pred_valid = model.predict(X_valid).reshape(-1, )
        #     score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
        #     print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
        #     print('')
        #     if X_test is not None:
        #         y_pred = model.predict_proba(X_test)
        #
        # if model_type == 'cat':
        #     raise Exception("Implement catboost Regression!")
        #
        #     model = CatBoostClassifier(iterations=n_estimators,
        #                                eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
        #                                loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
        #     model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
        #               verbose=False)
        #
        #     y_pred_valid = model.predict(X_valid)
        #     if X_test is not None:
        #         y_pred = model.predict(X_test)

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

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols

    return result_dict


