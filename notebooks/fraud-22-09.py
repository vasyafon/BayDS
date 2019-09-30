#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
import lightgbm as lgb
import numpy as np
import sys
# sys.path.append('..\Python Scripts\pipeline')
import warnings

warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import uniform
import datetime
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from scipy import stats
# from hyperopt import hp, tpe
# from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, validation_curve, \
    KFold
from sklearn.metrics import roc_auc_score
import seaborn as sns

import gc

# In[2]:

main_path = r'../..'
data_path = main_path + '/data/'

import sys

sys.path.append(main_path)
from BayDS.lib.training import *

c8 = pd.read_pickle(r'e:/kaggle/NewId\C8_agg.pkl')
c5 = pd.read_pickle(r'e:/kaggle/NewId\C5_agg.pkl')
tramt = pd.read_pickle(r'e:/kaggle/NewId\TransactionAmt_agg.pkl')

train = pd.read_pickle('e:/kaggle/data/train_09457_with_additions.pkl')
test = pd.read_pickle('e:/kaggle/data/test_09457_with_additions.pkl')

train = train.join(c8)
train = train.join(c5)
train = train.join(tramt)

test = test.join(tramt).join(c5).join(c8)
y = pd.read_pickle('e:/kaggle/data/y.pkl')

cols = "TransactionDT,TransactionAmt,ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,M1,M2,M3,M4,M5,M6,M7,M8,M9".split(
    ",")
train_test = train[cols].append(test[cols])

for col in "ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14".split(
        ","):
    col_count = train_test.groupby(col)['TransactionDT'].count()
    train[col + '_count'] = train[col].map(col_count)
    test[col + '_count'] = test[col].map(col_count)
#     print(col,test[col].map(lambda train:0 if train in s else 1).sum())


for col in "card1,card2,card5,addr1,addr2".split(","):
    col_count = train_test.groupby(col)['TransactionAmt'].mean()
    train[col + '_amtcount'] = train[col].map(col_count)
    test[col + '_amtcount'] = test[col].map(col_count)
    col_count1 = train_test[train_test['C5'] == 0].groupby(col)['C5'].count()
    col_count2 = train_test[train_test['C5'] != 0].groupby(col)['C5'].count()
    train[col + '_C5count'] = train[col].map(col_count2) / (train[col].map(col_count1) + 0.01)
    test[col + '_C5count'] = test[col].map(col_count2) / (test[col].map(col_count1) + 0.01)

del train_test
gc.collect()

train.drop([x for x in train.columns if x.startswith('DT')], axis=1, inplace=True)
test.drop([x for x in test.columns if x.startswith('DT')], axis=1, inplace=True)

train.drop([x for x in train.columns if 'by_card_id_ws' in x], axis=1, inplace=True)
test.drop([x for x in test.columns if 'by_card_id_ws' in x], axis=1, inplace=True)

# In[37]:


train.drop(['Date', 'TransactionDT'], axis=1, inplace=True)
test.drop(['Date', 'TransactionDT'], axis=1, inplace=True)

train = train.drop([x for x in train.columns if 'DT' in x], axis=1)
test = test.drop([x for x in test.columns if 'DT' in x], axis=1)

# In[39]:

newid = pd.read_pickle(r'e:/kaggle/NewId\\new_id.pkl')

train = train.join(newid)
test = test.join(newid)

train['start_date'] = pd.to_timedelta(pd.to_datetime(train['start_date'])).map(lambda x: x.days)
test['start_date'] = pd.to_timedelta(pd.to_datetime(test['start_date'])).map(lambda x: x.days)

timefreq = pd.read_pickle(r'e:/kaggle/TimeAggs24.09\transaction_frequencies.pkl')

train = train.join(timefreq)
test = test.join(timefreq)

train.drop('isFraud', axis=1, inplace=True)

print(train.shape, test.shape)

# ##### best submit 95.04: time freq, new card id, aggs tr amt, c5, c8; old 09457; all sample

# In[49]:
train.to_pickle('e:/kaggle/09504/train.pkl')
test.to_pickle('e:/kaggle/09504/test.pkl')
y.to_pickle('e:/kaggle/09504/y.pkl')

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          }

n_fold = 5
folds = KFold(n_splits=n_fold)
train_options = {
    "model_type": 'lgb',
    "params": params,
    "eval_metric": 'auc',
    'early_stopping_rounds': 200,
    'n_estimators': 10000,
    'averaging': 'usual',
    'use_groups': False,
    'fold_name': folds.__class__.__name__,
    'n_splits': n_fold}

result_timefreq = train_model_classification(X=train, X_test=test, y=y, params=params, folds=folds,
                                             model_type=train_options['model_type'],
                                             eval_metric=train_options['eval_metric'],
                                             plot_feature_importance=True,
                                             verbose=500, early_stopping_rounds=train_options['early_stopping_rounds'],
                                             n_estimators=train_options['n_estimators'],
                                             averaging=train_options['averaging'],
                                             n_jobs=-1, groups=None)

# In[50]:


pd.DataFrame(result_timefreq['oof'], columns=['isFraud'], index=train.index).to_csv(
    'oof_all_data_best_lgb_timefreq.csv')
pd.DataFrame(result_timefreq['prediction'], columns=['isFraud'], index=test.index).to_csv(
    'prediction_all_data_best_lgb_timefreq.csv')

sample_submission = pd.read_csv(data_path + 'sample_submission.csv').set_index('TransactionID')

sub1 = pd.DataFrame(result_timefreq['prediction'], columns=['isFraud'], index=test.index)

sample_submission['isFraud'] = sub1  # *0.5 + sub2*0.25 + sub3*0.25

sample_submission.to_csv('lgb_timefreq.csv')
