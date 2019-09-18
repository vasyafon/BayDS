#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Divide into X and y and scale
def data(df):
    ss = StandardScaler()

    X = df.drop(['target'], axis=1)
    y = df['target']

    X = pd.DataFrame(ss.fit_transform(X))
    X.columns = df.drop(['target'], axis=1).columns
    
    print('Data is normalized')
    print('There are %d objects and %d features' %(X.shape[0], X.shape[1]) )
    return X, y




# Plot distribution and skew+kurtosis
def distrib(df):
    plt.figure(figsize=(12, 8))
    sns.distplot(df['target'])
    print("Skewness: %f" % df['target'].skew())
    print("Kurtosis: %f" % df['target'].kurt())




# Differentiate numerical features (minus the target) and categorical features
def num_cat(df):
    categorical_features = df.select_dtypes(include = ["object"]).columns
    numerical_features = df.select_dtypes(exclude = ["object"]).columns
    if 'target' in numerical_features:
        numerical_features = numerical_features.drop('target')
    print("Numerical features : " + str(len(numerical_features)))
    print("Categorical features : " + str(len(categorical_features)))
    
    df_num = df[numerical_features]
    df_cat = df[categorical_features]
    return df_num, df_cat



# Correlation matrix
def corr_matrix(df, zoomed=True):
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    if zoomed:
        k=10
        plt.figure(figsize=(12, 8))
        cols = corrmat.nlargest(k, 'target')['target'].index
        print(cols)
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25) 
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
    return corrmat


# Scatterplot
def scatter_table(df, cols):
    sns.set()
    sns.pairplot(df[cols], size = 2.5)
    plt.show()



# Histogram and normal probability plot
def hist_norm(series):
    from scipy.stats import norm
    plt.figure(figsize=(12, 8))
    sns.distplot(series, fit=norm)
    fig = plt.figure()
    res = stats.probplot(series, plot=plt)
    
    
    
    
# Classification    
def kde_target(var_name, df):    
    corr = df['target'].corr(df[var_name])
    avg_repaid = df.loc[df['target'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['target'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 9))
    
    sns.kdeplot(df.loc[df['target'] == 0, var_name], label = 'target == 0')
    sns.kdeplot(df.loc[df['target'] == 1, var_name], label = 'target == 1')
    
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()
    
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)



# Relationship with numerical variables
def relation_numerical_target(df, var):
   
    data = pd.concat([df['target'], df[var]], axis=1)
    plt.figure(figsize=(12, 8))
    data.plot.scatter(x=var, y='target')




# Relationship with categorical features
def relation_categorical_target(df, var):
    
    data = pd.concat([df['target'], df[var]], axis=1)
    f, ax = plt.subplots(figsize=(12, 8))
    fig = sns.boxplot(x=var, y='target', data=data)
    fig.axis()

    
def count_values(df, col):
    temp = df[col].value_counts()
    df = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                      })
    plt.figure(figsize = (9,9))
    plt.title('%s values' %col)
    sns.set_color_codes("pastel")
    sns.barplot(x = 'labels', y="values", data=df)
    locs, labels = plt.xticks()
    plt.show()

