
# Общий (обработка категорий — label/ohe/frequency, проекция числовых на категории, трансформация числовых, бининг)
# Для регрессий (различное масштабирование)

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




def yesno(df, col):
    df[col] = df[col].map({'Y':1, 'N':0})
    return df




# def skewed(df):
#     from scipy.stats import skew
#     skewness = pd.DataFrame()
#     for feature in df.columns:
#         if df[feature].dtypes == "object":
#             print('need numerical features')
#         else:
#             skewed_feat = df[feature].apply(lambda x: skew(x)).sort_values(ascending=False)
#             print(skewed_feat)
#             skewness = skewness.append(skewed_feat)
# #         skewness = skewness[abs(skewness) > 0.75]
# #     print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#     return skewness





# Check the skew of all numerical features
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed

def box_cox_transform(df, feature, lam=0.15):
    from scipy.special import boxcox1p
#     lam = 0.15
    df[feature] = boxcox1p(df[feature], lam)
    return df
        
def log_transform(df, feature):
    df[feature] = np.log1p(df[feature])
    return df



def to_dummy(df):
    #convert categorical variable into dummy

    return df


def num_to_cat(df, feature):
    #transforming some numerical variables that are really categorical
    df[feature] = df[feature].apply(str)
    return df



# **Label Encoding some categorical variables that may contain information in their ordering set** 

def factorize(df, factor_df, column, fill_na=None):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    factor_df[column] = df[column]
    
    if fill_na is not None:
        factor_df[column].fillna(fill_na, inplace=True)
    le.fit(factor_df[column].unique())
    factor_df[column] = le.transform(factor_df[column])
    return factor_df


# Преобразование категориальных в числовые
#  при использовании этого подхода
# мы всегда должны быть уверены, что признак не может принимать неизвестных ранее значений.


def label_encoder(df, col):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df[col] = pd.Series(label_encoder.fit_transform(df[col]))
    return df



# Convert categorical features using one-hot encoding.
def onehot_encoder( df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=""+ column_name)
    df = df.join(dummies)
    df = df.drop([column_name], axis=1)
    return df


# ## Биннинг

def WOE_for_feature(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    good = (df.groupby('x')['y'].sum() + 0.5) / np.sum(df['y'])
    bad = (df.groupby('x')['y'].count() - df.groupby('x')['y'].sum() + 0.5) / (len(df['y']) - np.sum(df['y']))
    WOE = np.log(good / bad)
    WOE = pd.Series(WOE).to_dict()
    return df['x'].apply(lambda x: WOE.get(x)), WOE




# Функция рассчитывает Information Value для одной фичи
# INPUT:
#        x - признак
#        y - целевой признак
# OUTPUT:
#        Information Value
def Information_Value_for_feature(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    good = (df.groupby('x')['y'].sum() + 0.5) / np.sum(df['y'])
    bad = (df.groupby('x')['y'].count() - df.groupby('x')['y'].sum() + 0.5) / (len(df['y']) - np.sum(df['y']))
    WOE = np.log(good / bad)
    IV = (good - bad)*WOE
    return IV.sum()




# Функция, которая возвращает границы оптимального разбиения признака. 
# Разбиение оптимизируется с помощью построения решающего дерева для пары признак-таргет.
# Разбиение будет оптимальным в терминах WOE, т.к. при построении решающих правил в дереве происходит
# поиск границ, которые разбивают признак максимизируя Gini
# INPUT:
#        x_bondaries - фича для разбиения
#        y_bondaries - целевая переменная (таргет) 
# OUTPUT:
#        границы оптимального разбиения фичи

def get_bondaries_for_feature(x_bondaries, y_bondaries, max_depth, min_samples_leaf):
    # Сделаем поиск оптимальных параметров дерева по сетке
    # Задаем сетку параметров. Перебирать будем параметры "глубина дерева" и "количество объектов в листе"
    parameters = {'max_depth':[max_depth], 
                  'min_samples_leaf': [min_samples_leaf]}
    dtc = DecisionTreeClassifier(criterion='gini', random_state=17)
    # Для подбора параметров на кросс-валидации задаем стратифицированные разбиения на 5 фолдов
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    clf = GridSearchCV(dtc, parameters, scoring='roc_auc', cv=skf)
    clf.fit(pd.DataFrame(x_bondaries), y_bondaries)
    print('Best parameters for DT: ', clf.best_params_)
    print('ROC_AUC score: ', round(clf.best_score_, 4))
    tree = clf.best_estimator_
    # Обучаем дерево с наилучшими параметрами
    tree.fit(pd.DataFrame(x_bondaries), y_bondaries)
    print('Boundaries: ', np.sort([x for x in tree.tree_.threshold if x!=-2]))
    # Выгружаем границы оптимального разбиения
    return np.sort([x for x in tree.tree_.threshold if x!=-2]), clf.best_score_



# Функция считает границы разбиений для всех признаков используя get_bondaries_for_feature()
# INPUT:
#        df - набор признаков (без целевого)
#        y - целевой признак (таргет) 
# OUTPUT:
#        bondaries - границы оптимального разбиения фичи (словарь: название признака ключ).
#                    Список упорядоченный, это важно.
def get_bondaries_for_all_data(df, y, max_depth, min_samples_leaf):
    bondaries = dict()
    best_scores = dict()
    for column_name in df.columns:
        print (column_name)
        bondaries[column_name], best_scores[column_name] = get_bondaries_for_feature(df[column_name], y, max_depth, min_samples_leaf)
        print ('--------------')
    return bondaries, best_scores




# Функция каторая режет признак на интервалы и кодирует их
# Циклом пробегаем по границам признака и кодируем. Отрезок (-inf, bondaries[0]] кодируется нулём,
# дальше по возрастающей. Так же возвращает в текстовом виде границы кодируемого интервала
# INPUT:
#        x - значение признака
#        bondaries - границы разбиения данного признака
# OUTPUT:
#        код интервала в который попадает данное значение признака
def feature_splitter_encoder(x, bondaries):
    for i in range(len(bondaries)):
        if i>0:
            if x>bondaries[i-1] and x<=bondaries[i]:
                return i, bondaries[i-1], bondaries[i]
        if i==0:
            if x<=bondaries[i]:
                return i, float('-inf'), bondaries[i]
        if i==len(bondaries)-1:
            if x>bondaries[i]:
                return i+1, bondaries[i], float('inf')



# Функция нарезает все признаки на интервалы, кодирует их и считает WOE
# INPUT:
#        df - набор признаков (без целевого)
#        y - целевой признак (таргет) 
# OUTPUT:
#        df_woe - набор признаков закодированных WOE
#        data_all - набор признаков + WOE + коды интервалов + интервалы
#        bondaries - границы разбиений

def splitter(df, y, max_depth=4, min_samples_leaf=50):
    # Считаем границы разбиений
    bondaries, best_scores = get_bondaries_for_all_data(df, y, max_depth, min_samples_leaf)
    df_woe = pd.DataFrame()
    list_of_const_columns = []
    df_woe_borders = pd.DataFrame()
    # Для каждого признака считаем WOE
    woe_table = dict()
    decode = dict()
    for column_name in df.columns:
        df_woe_borders = pd.DataFrame()
        # Исходные значения признаков
        df_woe_borders[column_name + '_value'] = df[column_name]
        # Создаем разбиения и кодируем их
        df_woe[column_name] = df[column_name].apply(lambda x: feature_splitter_encoder(x, bondaries[column_name])[0])
        # Записываем границы разбиений 
        df_woe_borders[column_name + '_min'] = df[column_name].apply(lambda x: feature_splitter_encoder(x, bondaries[column_name])[1])
        df_woe_borders[column_name + '_max'] = df[column_name].apply(lambda x: feature_splitter_encoder(x, bondaries[column_name])[2])
        
        if np.mean(df_woe[column_name]) == np.max(df_woe[column_name]):
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print('ATTENTION !!!')
            print('Column "' + column_name + '" is constant!')
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        list_of_const_columns.append(column_name)
        
        # Считаем WOE
        df_woe[column_name], decode[column_name] = WOE_for_feature(df_woe[column_name], y)
        df_woe_borders[column_name + '_WOE'] = df_woe[column_name]
        df_woe_borders = df_woe_borders[[column_name + '_min', column_name + '_max', column_name + '_WOE']]
        df_woe_borders = df_woe_borders.drop_duplicates()
        df_woe_borders = df_woe_borders.sort_values([column_name + '_min'])
        df_woe_borders = df_woe_borders.reset_index(drop=True)
        woe_table[column_name] = df_woe_borders
        
    return df_woe, woe_table, bondaries, decode




# Считаем Information Value для всего датасета
# INPUT:
#        df - набор признаков (без целевого) закодированный WOE (!)
#        y - целевой признак (таргет) 
# OUTPUT:
#        IVs - набор значений Information Value для каждого признака (словарь)
def Information_Value(df, y):
    IVs = dict()
    for column_name in df.columns:
        IVs[column_name] = Information_Value_for_feature(df[column_name], y)
        
    iv_importances = list(IVs.values())
    iv_importances = np.array(iv_importances)
    iv_indices = np.argsort(iv_importances)[::-1]

    plt.figure(figsize=(15, 10))
    plt.title("Feature importances")
    plt.bar(range(df.shape[1]), iv_importances[list(iv_indices)],
       color="r")
    plt.xticks(range(len(df.columns[iv_indices])), df.columns[iv_indices], rotation='vertical')
    plt.show()

    return IVs


## Time


def dates(df, date_cols):
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
#         df['year'] = df['first_active_month'].dt.year
#         df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (datetime.date(2018, 2, 1) - df[col].dt.date).dt.days
    return df

