def quick_score(data, categorial_features, colname=None, replacement=None, target='isFraud', drop_columns=['Date'], filter_neg_target=True,
                isReplacementCategorial=False, params=None, n_estimators = 6):
    if params is None:
        params = {'num_leaves': 500,
                  'min_child_weight': 0.03454472573214212,
                  'feature_fraction': 0.3797454081646243,
                  'bagging_fraction': 0.4181193142567742,
                  'min_data_in_leaf': 106,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.1,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 0.3899927210061127,
                  'reg_lambda': 0.6485237330340494,
                  'random_state': 47,
                  }

    if filter_neg_target:
        train_subset_ids = data[data[target] >= 0].index
    else:
        train_subset_ids = data.index

    X_new = data.loc[train_subset_ids].drop(drop_columns + [target], axis=1)
    if colname is not None:
        assert replacement is not None
        X_new[colname] = replacement.loc[train_subset_ids]
    y = data.loc[train_subset_ids][target]

    folds = KFold(n_splits=5, shuffle=False)

    if isReplacementCategorial:
        raise Exception("Add replacement column as categorial")

    categorical_columns = [c for c, col in enumerate(X_new.columns) if col in categorial_features]

    params['categorical_feature'] = categorical_columns

    results = train_model_classification(X=X_new, X_test=None, y=y, params=params, folds=folds, splits=1,
                                         model_type='lgb', eval_metric='auc', plot_feature_importance=False,
                                         verbose=None, early_stopping_rounds=40, n_estimators=600, averaging='usual',
                                         n_jobs=-1)

    return sum(results['scores']) / len(results['scores'])

