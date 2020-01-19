from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
import pandas as pd


def individual_gini(feat, df, y, n_jobs):
    '''
    count individual Gini for feature
    :param feat: feature to count Gini for
    :param df: dataframe
    :param y: target series
    :param n_jobs: n_jobs
    :return: individual Gini
    '''
    #     print(' feature: ', feat)
    df = df[feat]
    if df.dtypes == 'O':
        x = pd.get_dummies(df).values
        obvious_gini = 0
    else:
        #         print('\t  Counting obvious gini...')
        if df.dtype.name in ('float32', 'float64'):
            df = df.replace({np.inf: np.nan, -np.inf: np.nan})
            df = df.fillna(-99999)
        x = np.array(df.values).reshape(-1, 1)
        obvious_gini = round(abs(roc_auc_score(y, x) * 2 - 1), 3)

    #     print('\t  Now count gini using decision tree')
    parameters = {'min_weight_fraction_leaf': [0.01, 0.025, 0.05, 0.1]}

    dt = DecisionTreeClassifier(random_state=0)
    cv = StratifiedKFold(3, random_state=0)
    clf = GridSearchCV(dt, parameters, cv=cv, scoring='roc_auc', n_jobs=n_jobs)

    #     print('\t  Run grid search...')
    clf.fit(x, y)

    true_gini = round(abs(clf.best_score_ * 2 - 1), 3)
    #     print('\t  True gini is ', true_gini)
    #     print('-------------------------------------------')

    return max(100 * obvious_gini, 100 * true_gini)
