from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from .node import Node
from .pipeline import *
from ..io import *
import numpy as np
import os
from sklearn import preprocessing
from ..encoding import LabelEncoderPopularity
from ..aggregations.temporal import aggregate_with_time_local, aggregate_transaction_frequencies
import multiprocess as mp


class IEEEFraudTransactionLoaderNode(Node):
    params = {
        'input_directory': None
    }

    def _run(self):
        folder_path = self.params['input_directory']
        files = [f'{folder_path}/train_transaction.csv',
                 f'{folder_path}/test_transaction.csv']
        train_transaction, test_transaction = map(load_data, files)
        test_transaction['isFraud'] = -1
        data = pd.concat([train_transaction, test_transaction], axis=0, sort=False)
        data.set_index('TransactionID', inplace=True)
        self.output = data


class IEEEFraudIdentityLoaderNode(Node):
    params = {
        'input_directory': None
    }

    def _run(self):
        folder_path = self.params['input_directory']
        files = [f'{folder_path}/train_identity.csv',
                 f'{folder_path}/test_identity.csv']
        train_identity, test_identity = map(load_data, files)
        data_id = pd.concat([train_identity, test_identity], axis=0, sort=False)
        data_id.set_index('TransactionID', inplace=True)
        self.output = data_id


class AddNaNCountNode(Node):
    params = {
        'name': None,
        'inplace': True
    }

    def _run(self):
        nan_count = self.input.isna().sum(axis=1)
        if self.params['inplace']:
            self.input[self.params['name']] = nan_count
            self.output = self.input
        else:
            self.output = pd.DataFrame(index=self.input.index)
            self.output[self.params['name']] = nan_count


class TimeTransformNode(Node):
    '''
    Removes TransactionDT and adds Date variable + weekday + hours + days
    '''

    def _run(self):
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        START_DATE = '2017-12-01'
        startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

        data = self.input
        data["Date"] = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        data['Weekdays'] = data['Date'].dt.dayofweek
        data['Hours'] = data['Date'].dt.hour
        data['Days'] = data['Date'].dt.day

        # Taken from Anna Notebook
        data['isNight'] = data['Hours'].map(lambda x: 1 if (x >= 23 or x < 5) else 0)
        data['DT_M'] = (data['Date'].dt.year - 2017) * 12 + data['Date'].dt.month
        data['DT_W'] = (data['Date'].dt.year - 2017) * 52 + data['Date'].dt.weekofyear
        data['DT_D'] = (data['Date'].dt.year - 2017) * 365 + data['Date'].dt.dayofyear

        data['is_december'] = data['Date'].dt.month
        data['is_december'] = (data['is_december'] == 12).astype(np.int8)
        # dataset.drop(['TransactionDT'], axis=1, inplace=True)

        dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

        data['is_holiday'] = (data['Date'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)


class AnyaNewFENode(Node):
    def _run(self):
        data = self.input
        data['TransactionAmt_Log'] = np.log(data['TransactionAmt'])
        data['TransactionAmt_Log1p'] = np.log1p(data['TransactionAmt'])
        data['TransactionAmt'] = data['TransactionAmt'].clip(0, 5000)
        data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(
            int)
        data['nulls1'] = data.isna().sum(axis=1)

        # Transform D pipeline
        for col in ['D' + str(i) for i in range(1, 16)]:
            data[col] = data[col].clip(0)

        data['D9_not_na'] = np.where(data['D9'].isna(), 0, 1)
        data['D8_not_same_day'] = np.where(data['D8'] >= 1, 1, 0)
        data['D8_D9_decimal_dist'] = data['D8'].fillna(0) - data['D8'].fillna(0).astype(int)
        data['D8_D9_decimal_dist'] = np.abs(data['D8_D9_decimal_dist'] - data['D9'])
        data['D8'] = data['D8'].fillna(-1).astype(int)

        for col in ['D1', 'D2']:
            data[col + '_scaled'] = data[col] / data[data['isFraud'] != -1][col].max()

        # Other stuff
        data['M_sum'] = data[['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']].sum(axis=1).astype(np.int8)
        data['M_na'] = data[['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']].isna().sum(axis=1).astype(np.int8)

        a = np.zeros(data.shape[0])
        data["lastest_browser"] = a

        def browser(df):
            df.loc[df["id_31"] == "samsung browser 7.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "opera 53.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "mobile safari 10.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "google search application 49.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "firefox 60.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "edge 17.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 69.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 67.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for ios", 'lastest_browser'] = 1
            return df

        browser(data)

        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0].astype(str)
            dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1].astype(str)

            dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1].astype(str)

            dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1].astype(str)

            dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype(np.int32)
            dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype(np.int32)

            dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1].astype(str)
            dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1].astype(str)

            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                         dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            dataframe['had_id'] = 1
            gc.collect()

        # Some arbitrary features interaction
        for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain',
                        'P_emaildomain__C2',
                        'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
            f1, f2 = feature.split('__')
            data[feature] = data[f1].astype(str) + '_' + data[f2].astype(str)

            le = preprocessing.LabelEncoder()
            le.fit(list(data[feature].astype(str).values))
            data[feature] = le.transform(list(data[feature].astype(str).values))

        # ??????????? ? RARE CARD
        train_df = data[data['isFraud'] != -1]
        test_df = data[data['isFraud'] == -1]

        for col in ['card1']:
            valid_card = data[[col]]  # pd.concat([train_df[[col]], test_df[[col]]])
            valid_card = valid_card[col].value_counts()
            valid_card_std = valid_card.values.std()

            invalid_cards = valid_card[valid_card <= 2]

            valid_card = valid_card[valid_card > 2]
            valid_card = list(valid_card.index)

            train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
            test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

            train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
            test_df[col] = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

        for col in ['card2', 'card3', 'card4', 'card5', 'card6', ]:
            print('No intersection in Train', col, len(train_df[~train_df[col].isin(test_df[col])]))
            print('Intersection in Train', col, len(train_df[train_df[col].isin(test_df[col])]))

            train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
            test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

        train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
        test_df['TransactionAmt_check'] = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

        for df in [train_df, test_df]:
            for col in ['C' + str(i) for i in range(1, 15)]:
                max_value = train_df[train_df['DT_M'] == train_df['DT_M'].max()][col].max()
                df[col] = df[col].clip(None, max_value)

        data = pd.concat([train_df, test_df])

        # MAKE UIDS
        data['uid'] = data['card1'].astype(str) + '_' + data['card2'].astype(str)

        data['uid2'] = data['uid'].astype(str) + '_' + data['card3'].astype(str) + '_' + data['card4'].astype(str)

        data['uid3'] = data['uid2'].astype(str) + '_' + data['addr1'].astype(str) + '_' + data['addr2'].astype(str)

        data['uid4'] = data['uid3'].astype(str) + '_' + data['P_emaildomain'].astype(str)

        data['uid5'] = data['uid3'].astype(str) + '_' + data['R_emaildomain'].astype(str)

        data['bank_type'] = data['card3'].astype(str) + '_' + data['card5'].astype(str)

        data['product_type'] = data['ProductCD'].astype(str) + '_' + data['TransactionAmt'].astype(str)

        for col in ['ProductCD', 'M4']:
            temp_dict = data.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(
                columns={'mean': col + '_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col + '_target_mean'].to_dict()

            data[col + '_target_mean'] = data[col].map(temp_dict)

        self.output = data

class AnyaFinalFENode(Node):
    def _run(self):
        data = self.input

        for col in "card1,card2,card5,addr1,addr2".split(","):
            col_count = data.groupby(col)['TransactionAmt'].mean()
            data[col + '_amtcount'] = data[col].map(col_count)
            col_count1 = data[data['C5'] == 0].groupby(col)['C5'].count()
            col_count2 = data[data['C5'] != 0].groupby(col)['C5'].count()
            data[col + '_C5count'] = data[col].map(col_count2) / (data[col].map(col_count1) + 0.01)


class SomeAggregatesFromAnyaNode(Node):
    def _run(self):
        data = self.input
        data['id_02_to_mean_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('mean')
        data['id_02_to_mean_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('mean')
        data['id_02_to_std_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('std')
        data['id_02_to_std_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('std')

        data['D15_to_mean_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('mean')
        data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')
        data['D15_to_std_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('std')
        data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')

        data['D15_to_mean_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('mean')
        data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')
        data['D15_to_std_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('std')
        data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')

        data['TransactionAmt_to_mean_card1'] = data['TransactionAmt'] / data.groupby(['card1'])[
            'TransactionAmt'].transform('mean')
        data['TransactionAmt_to_mean_card4'] = data['TransactionAmt'] / data.groupby(['card4'])[
            'TransactionAmt'].transform('mean')
        data['TransactionAmt_to_std_card1'] = data['TransactionAmt'] / data.groupby(['card1'])[
            'TransactionAmt'].transform('std')
        data['TransactionAmt_to_std_card4'] = data['TransactionAmt'] / data.groupby(['card4'])[
            'TransactionAmt'].transform('std')

        data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(
            int)
        data['nulls1'] = data.isna().sum(axis=1)

        a = np.zeros(data.shape[0])
        data["lastest_browser"] = a

        def browser(df):
            df.loc[df["id_31"] == "samsung browser 7.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "opera 53.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "mobile safari 10.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "google search application 49.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "firefox 60.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "edge 17.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 69.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 67.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for ios", 'lastest_browser'] = 1
            return df

        browser(data)

        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0].astype(str)
            dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1].astype(str)

            dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1].astype(str)

            dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1].astype(str)

            dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype(np.int32)
            dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype(np.int32)

            dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1].astype(str)
            dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1].astype(str)

            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                         dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            dataframe['had_id'] = 1
            gc.collect()

        id_split(data)

        data['TransactionAmt_Log'] = np.log(data['TransactionAmt'])

        data['card1_count_full'] = data['card1'].map(data['card1'].value_counts(dropna=False))

        # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
        data['Transaction_day_of_week'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 7)
        data['Transaction_hour'] = np.floor(data['TransactionDT'] / 3600) % 24

        # Some arbitrary features interaction
        for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain',
                        'P_emaildomain__C2',
                        'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
            f1, f2 = feature.split('__')
            data[feature] = data[f1].astype(str) + '_' + data[f2].astype(str)

            le = preprocessing.LabelEncoder()
            le.fit(list(data[feature].astype(str).values))
            data[feature] = le.transform(list(data[feature].astype(str).values))

        useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1',
                           'addr2', 'dist1',
                           'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
                           'C11',
                           'C12', 'C13',
                           'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14',
                           'D15',
                           'M2', 'M3',
                           'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                           'V12', 'V13', 'V17',
                           'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45',
                           'V46',
                           'V47', 'V48',
                           'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64',
                           'V69',
                           'V70', 'V71',
                           'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87',
                           'V90',
                           'V91', 'V92',
                           'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131',
                           'V138',
                           'V139', 'V140',
                           'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158',
                           'V159',
                           'V160', 'V161',
                           'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173',
                           'V175',
                           'V176', 'V177',
                           'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201',
                           'V202',
                           'V203', 'V204',
                           'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216',
                           'V217',
                           'V219', 'V220',
                           'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233',
                           'V234',
                           'V238', 'V239',
                           'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257',
                           'V258',
                           'V259', 'V261',
                           'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273',
                           'V274',
                           'V275', 'V276',
                           'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291',
                           'V292',
                           'V294', 'V303',
                           'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322',
                           'V323',
                           'V324', 'V326',
                           'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05',
                           'id_06', 'id_09',
                           'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31',
                           'id_32', 'id_33',
                           'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

        # for feature in ['id_34', 'id_36']:
        #     if feature in useful_features:
        #         # Count encoded for both train and test
        #         train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        #         test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:
            if feature in useful_features:
                # Count encoded separately for train and test
                data[feature + '_count_dist'] = data[feature].map(data[feature].value_counts(dropna=False))

        data['card1_count_full'] = data['card1'].map(data['card1'].value_counts(dropna=False))
        data['card2_count_full'] = data['card2'].map(data['card2'].value_counts(dropna=False))
        data['card3_count_full'] = data['card3'].map(data['card3'].value_counts(dropna=False))
        data['card4_count_full'] = data['card4'].map(data['card4'].value_counts(dropna=False))
        data['card5_count_full'] = data['card5'].map(data['card5'].value_counts(dropna=False))
        data['card6_count_full'] = data['card6'].map(data['card6'].value_counts(dropna=False))
        data['addr1_count_full'] = data['addr1'].map(data['addr1'].value_counts(dropna=False))
        data['addr2_count_full'] = data['addr2'].map(data['addr2'].value_counts(dropna=False))

        category_features = ["ProductCD", "P_emaildomain",
                             "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType",
                             "DeviceInfo", "id_12",
                             "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22",
                             "id_23", "id_24",
                             "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_32", "id_34", 'id_36'
                                                                                                     "id_37", "id_38"]
        for c in category_features:
            data[feature + '_count_full'] = data[feature].map(data[feature].value_counts(dropna=False))

        i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

        data['M_sum'] = data[i_cols].sum(axis=1).astype(np.int8)
        data['M_na'] = data[i_cols].isna().sum(axis=1).astype(np.int8)

        for col in ['ProductCD', 'M4']:
            temp_dict = data.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(
                columns={'mean': col + '_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col + '_target_mean'].to_dict()

            data[col + '_target_mean'] = data[col].map(temp_dict)

        # todo: find out if it required
        # data['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)

        data['uid'] = data['card1'].astype(str) + '_' + data['card2'].astype(str)

        data['uid2'] = data['uid'].astype(str) + '_' + data['card3'].astype(str) + '_' + data['card4'].astype(str)

        data['uid3'] = data['uid2'].astype(str) + '_' + data['addr1'].astype(str) + '_' + data['addr2'].astype(str)

        i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

        for col in i_cols:
            for agg_type in ['mean', 'std']:
                new_col_name = col + '_TransactionAmt_' + agg_type
                temp_df = data[[col, 'TransactionAmt']]
                # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                data[new_col_name] = data[col].map(temp_df)

        i_cols = ['card1', 'card2', 'card3', 'card5',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                  'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                  'addr1', 'addr2',
                  'dist1', 'dist2',
                  'P_emaildomain', 'R_emaildomain',
                  'DeviceInfo',
                  'id_30',
                  #           'id_30_device',
                  'version_id_30',
                  'version_id_31',
                  'id_33',
                  'uid', 'uid2', 'uid3',
                  ]

        for col in i_cols:
            temp_df = data[[col]]
            fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
            data[col + '_fq_enc'] = data[col].map(fq_encode)

        for col in ['DT_M', 'DT_W', 'DT_D']:
            temp_df = data[[col]]
            fq_encode = temp_df[col].value_counts().to_dict()

            data[col + '_total'] = data[col].map(fq_encode)

        periods = ['DT_M', 'DT_W', 'DT_D']
        i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']
        for period in periods:
            for col in i_cols:
                new_column = col + '_' + period

                temp_df = data[[col, period]]
                temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
                fq_encode = temp_df[new_column].value_counts().to_dict()

                data[new_column] = (data[col].astype(str) + '_' + data[period].astype(str)).map(fq_encode)
                data[new_column] /= data[period + '_total']


class SomeAggregatesFromAnyaNewCardIdNode(Node):
    def _run(self):
        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]

        count_full = ['new_card_id', 'addr1', 'addr2'] + [f'card{i}' for i in range(1, 7)] + \
                     ["ProductCD", "P_emaildomain",
                      "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType",
                      "DeviceInfo", 'id_01', "id_12",
                      "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22",
                      "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", 'id_31', "id_32",
                      'id_33', "id_34", 'id_36', "id_37", "id_38"]
        features_interaction = ['new_card_id__dist1',
                                'new_card_id__id_20',
                                'new_card_id__P_emaildomain',
                                'new_card_id__addr1',
                                'P_emaildomain__C2',
                                'id_02__id_20',
                                'id_02__D8',
                                'D11__DeviceInfo',
                                'DeviceInfo__P_emaildomain']

        data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(
            int)
        num_cols.append('TransactionAmt_decimal')

        data['nulls1'] = data.isna().sum(axis=1)
        num_cols.append('nulls1')

        a = np.zeros(data.shape[0])
        data["lastest_browser"] = a

        def browser(df):
            df.loc[df["id_31"] == "samsung browser 7.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "opera 53.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "mobile safari 10.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "google search application 49.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "firefox 60.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "edge 17.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 69.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 67.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 63.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 64.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 65.0 for ios", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for android", 'lastest_browser'] = 1
            df.loc[df["id_31"] == "chrome 66.0 for ios", 'lastest_browser'] = 1
            return df

        browser(data)
        cat_cols.append('lastest_browser')

        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0].astype(str)
            dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1].astype(str)

            dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1].astype(str)

            dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0].astype(str)
            dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1].astype(str)

            dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype(np.float32)
            dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype(np.float32)

            dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1].astype(str)
            dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1].astype(str)

            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                         dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            dataframe['had_id'] = 1
            gc.collect()

        id_split(data)
        cat_cols.append('device_name')
        cat_cols.append('device_version')
        cat_cols.append('OS_id_30')
        cat_cols.append('version_id_30')
        cat_cols.append('version_id_31')
        cat_cols.append('browser_id_31')
        num_cols.append('screen_width')
        num_cols.append('screen_height')
        cat_cols.append('had_id')

        data['TransactionAmt_Log'] = np.log(data['TransactionAmt'])
        num_cols.append('TransactionAmt_Log')

        # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
        data['Transaction_day_of_week'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 7)
        num_cols.append('Transaction_day_of_week')

        data['Transaction_hour'] = np.floor(data['TransactionDT'] / 3600) % 24
        num_cols.append('Transaction_hour')

        # Some arbitrary features interaction
        for feature in features_interaction:
            f1, f2 = feature.split('__')
            data[feature] = data[f1].astype(str) + '_' + data[f2].astype(str)
            le = preprocessing.LabelEncoder()
            le.fit(list(data[feature].astype(str).values))
            data[feature] = le.transform(list(data[feature].astype(str).values))
            cat_cols.append(feature)

        # useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1',
        #                    'addr2', 'dist1',
        #                    'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
        #                    'C11',
        #                    'C12', 'C13',
        #                    'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14',
        #                    'D15',
        #                    'M2', 'M3',
        #                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
        #                    'V12', 'V13', 'V17',
        #                    'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45',
        #                    'V46',
        #                    'V47', 'V48',
        #                    'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64',
        #                    'V69',
        #                    'V70', 'V71',
        #                    'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87',
        #                    'V90',
        #                    'V91', 'V92',
        #                    'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131',
        #                    'V138',
        #                    'V139', 'V140',
        #                    'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158',
        #                    'V159',
        #                    'V160', 'V161',
        #                    'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173',
        #                    'V175',
        #                    'V176', 'V177',
        #                    'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201',
        #                    'V202',
        #                    'V203', 'V204',
        #                    'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216',
        #                    'V217',
        #                    'V219', 'V220',
        #                    'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233',
        #                    'V234',
        #                    'V238', 'V239',
        #                    'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257',
        #                    'V258',
        #                    'V259', 'V261',
        #                    'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273',
        #                    'V274',
        #                    'V275', 'V276',
        #                    'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291',
        #                    'V292',
        #                    'V294', 'V303',
        #                    'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322',
        #                    'V323',
        #                    'V324', 'V326',
        #                    'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05',
        #                    'id_06', 'id_09',
        #                    'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31',
        #                    'id_32', 'id_33',
        #                    'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

        for feature in count_full:
            # Count encoded separately for train and test
            data[feature + '_count_full'] = data[feature].map(data[feature].value_counts(dropna=False))
            num_cols.append(feature + '_count_full')

        i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

        data['M_sum'] = data[i_cols].sum(axis=1).astype(np.int8)
        num_cols.append('M_sum')
        data['M_na'] = data[i_cols].isna().sum(axis=1).astype(np.int8)
        num_cols.append('M_na')

        for col in ['ProductCD', 'M4']:
            temp_dict = data.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(
                columns={'mean': col + '_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col + '_target_mean'].to_dict()
            data[col + '_target_mean'] = data[col].map(temp_dict)
            num_cols.append(col + '_target_mean')

        # todo: find out if it required
        # data['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)

        i_cols = ['card1', 'card2', 'card3', 'card5',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                  'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                  'addr1', 'addr2',
                  'dist1', 'dist2',
                  'P_emaildomain', 'R_emaildomain',
                  'DeviceInfo',
                  'id_30',
                  'version_id_30',
                  'version_id_31',
                  'id_33', 'new_card_id'
                  ]

        for col in i_cols:
            temp_df = data[[col]]
            fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
            data[col + '_fq_enc'] = data[col].map(fq_encode)
            num_cols.append(col + '_fq_enc')

    # numerical_cols = ['id_%02d' % i for i in range(1,12)] + ["V%d"%i for i in range(1,340)] + ["D%d"%i for i in range(1,16)] + ["C%d"%i for i in range(1,15)]  + ['dist1','TransactionAmt', 'NanIdentityCount', 'NanTransactionCount', '_Weekdays', '_Hours', '_Days', 'Date', 'dist2']
    # label_cols = ['M1', 'M2', 'M3','M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'card4', 'card6', 'ProductCD'] + ['id_%02d'%i for i in (12,15,16,28,29,32,34,35,36,37,38)]
    # label_cols += ['id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
    #             'id_30', 'id_31',  'id_33', 'DeviceType', 'DeviceInfo', 'P_emaildomain',
    #             'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2',
    #             'P_emaildomain_bin', 'P_emaildomain_suffix', 'R_emaildomain_bin', 'R_emaildomain_suffix']
    #
    # #add_from Anya Features
    # numerical_cols+=['TransactionDT', 'Weekdays', 'Hours', 'Days','DT_M', 'DT_W', 'DT_D','id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1', 'id_02_to_std_card4', 'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4', 'D15_to_mean_addr1', 'D15_to_std_addr1', 'TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4', 'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4', 'TransactionAmt_decimal','nulls1','screen_width', 'screen_height','TransactionAmt_Log', 'card1_count_full',   'Transaction_day_of_week', 'Transaction_hour', 'id_01_count_dist', 'id_31_count_dist', 'id_33_count_dist', 'id_36_count_dist', 'card2_count_full', 'card3_count_full', 'card4_count_full', 'card5_count_full', 'card6_count_full', 'addr1_count_full', 'addr2_count_full', 'id_36_count_full', 'M_sum', 'M_na', 'ProductCD_target_mean', 'M4_target_mean',  'card1_TransactionAmt_mean', 'card1_TransactionAmt_std', 'card2_TransactionAmt_mean', 'card2_TransactionAmt_std', 'card3_TransactionAmt_mean', 'card3_TransactionAmt_std', 'card5_TransactionAmt_mean', 'card5_TransactionAmt_std', 'uid_TransactionAmt_mean', 'uid_TransactionAmt_std', 'uid2_TransactionAmt_mean', 'uid2_TransactionAmt_std', 'uid3_TransactionAmt_mean', 'uid3_TransactionAmt_std', 'card1_fq_enc', 'card2_fq_enc', 'card3_fq_enc', 'card5_fq_enc', 'C1_fq_enc', 'C2_fq_enc', 'C3_fq_enc', 'C4_fq_enc', 'C5_fq_enc', 'C6_fq_enc', 'C7_fq_enc', 'C8_fq_enc', 'C9_fq_enc', 'C10_fq_enc', 'C11_fq_enc', 'C12_fq_enc', 'C13_fq_enc', 'C14_fq_enc', 'D1_fq_enc', 'D2_fq_enc', 'D3_fq_enc', 'D4_fq_enc', 'D5_fq_enc', 'D6_fq_enc', 'D7_fq_enc', 'D8_fq_enc', 'addr1_fq_enc', 'addr2_fq_enc', 'dist1_fq_enc', 'dist2_fq_enc', 'P_emaildomain_fq_enc', 'R_emaildomain_fq_enc', 'DeviceInfo_fq_enc', 'id_30_fq_enc', 'version_id_30_fq_enc', 'version_id_31_fq_enc', 'id_33_fq_enc', 'uid_fq_enc', 'uid2_fq_enc', 'uid3_fq_enc', 'DT_M_total', 'DT_W_total', 'DT_D_total', 'card1_DT_M', 'card2_DT_M', 'card3_DT_M', 'card5_DT_M', 'uid_DT_M', 'uid2_DT_M', 'uid3_DT_M', 'card1_DT_W', 'card2_DT_W', 'card3_DT_W', 'card5_DT_W', 'uid_DT_W', 'uid2_DT_W', 'uid3_DT_W', 'card1_DT_D', 'card2_DT_D', 'card3_DT_D', 'card5_DT_D', 'uid_DT_D', 'uid2_DT_D', 'uid3_DT_D']
    # label_cols+=['isNight','lastest_browser', 'device_name', 'device_version', 'OS_id_30', 'version_id_30', 'browser_id_31', 'version_id_31', 'had_id', 'id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1','uid', 'uid2', 'uid3']
    # strange_cols = ['Transaction_day_of_week', 'Transaction_hour']

    # p.data['numerical_columns'] = numerical_cols
    # p.data['categorical_columns'] = label_cols
    # p.data['useless_columns'] = strange_cols




class AddAggregatesTotalNode(Node):
    params = {
        'features': [],
        'group_by': 'card_id'
    }

    def _run(self):
        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        group_by_feature = self.params['group_by']

        for fname in self.params['features']:
            data[f'{fname}_to_mean_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('mean').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            data[f'{fname}_to_std_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('std').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            num_cols.extend([f'{fname}_to_mean_{group_by_feature}', f'{fname}_to_std_{group_by_feature}'])

        self.output = [data, num_cols, cat_cols]


class AddGroupNumericalAggregatesNode(Node):
    params = {
        'features': [],
        'group_by': ['new_card_id'],
        'to_mean': True,
        'to_std': True,
        'to_minmax': True,
        'to_std_score': True
    }

    def _run(self):
        df = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        groupby = self.params['group_by']
        cols = self.params['features']
        to_mean = self.params['to_mean']
        to_std = self.params['to_std']
        to_minmax = self.params['to_minmax']
        to_std_score = self.params['to_std_score']

        slice_df = df[groupby + cols]
        if len(groupby) == 1:
            gcname = groupby[0]
        else:
            gcname = "(" + '+'.join(groupby) + ")"

        for igc, gc in enumerate(groupby):
            if igc == 0:
                slice_df['group_col'] = slice_df[gc].astype(str)
            else:
                slice_df['group_col'] += '_' + slice_df[gc].astype(str)

        #     if freq:
        #         count = slice_df['group_col'].map(slice_df['group_col'].value_counts(dropna=False))

        for col in cols:
            group_col = slice_df.groupby(['group_col'])[col]
            dmean = group_col.transform('mean')
            dstd = group_col.transform('std').replace(-np.inf, np.nan).replace(np.inf, np.nan)

            if to_mean:
                df[f'{col}_to_mean_groupby_{gcname}'] = (slice_df[col] / dmean).astype(np.float32)
                num_cols.append(f'{col}_to_mean_groupby_{gcname}')

            if to_std:
                df[f'{col}_to_std_groupby_{gcname}'] = (slice_df[col] / dstd).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)
                num_cols.append(f'{col}_to_std_groupby_{gcname}')

            if to_minmax:
                dmin = group_col.transform('min').astype(np.float32)
                dmax = group_col.transform('max').astype(np.float32)
                df[f'{col}_to_minmax_groupby_{gcname}'] = ((slice_df[col] - dmin) / (dmax - dmin)).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)
                num_cols.append(f'{col}_to_minmax_groupby_{gcname}')

            if to_std_score:
                df[f'{col}_to_stdscore_groupby_{gcname}'] = ((slice_df[col] - dmean) / dstd).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)
                num_cols.append(f'{col}_to_stdscore_groupby_{gcname}')

        self.output = [df, num_cols, cat_cols]


class AddGroupFrequencyEncodingNode(Node):
    params = {
        'features': [],
        'group_by': ['new_card_id'],
        'count': True,
        'freq': False
    }

    def _run(self):
        df = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        groupby = self.params['group_by']
        cols = self.params['features']
        to_count = self.params['count']
        to_freq = self.params['freq']

        slice_df = df[groupby + cols]
        if len(groupby) == 1:
            gcname = groupby[0]
        else:
            gcname = "(" + '+'.join(groupby) + ")"

        for igc, gc in enumerate(groupby):
            if igc == 0:
                slice_df['group_col'] = slice_df[gc].astype(str)
            else:
                slice_df['group_col'] += '_' + slice_df[gc].astype(str)
        if to_freq:
            count = slice_df['group_col'].map(slice_df['group_col'].value_counts(dropna=False))
        for col in cols:
            tempdf = (slice_df['group_col'] + '_' + slice_df[col].astype(str))
            tempdf = tempdf.map(tempdf.value_counts(dropna=False))
            if to_count:
                df[f'{col}_count_groupby_{gcname}'] = tempdf
                num_cols.append(f'{col}_count_groupby_{gcname}')
            if to_freq:
                df[f'{col}_freq_groupby_{gcname}'] = tempdf / count
                num_cols.append(f'{col}_freq_groupby_{gcname}')

        self.output = [df, num_cols, cat_cols]


class AddGlobalFrequencyEncodingNode(Node):
    params = {
        'features': [],
        'count': True,
        'freq': False
    }

    def _run(self):
        df = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        cols = self.params['features']
        to_count = self.params['count']
        to_freq = self.params['freq']

        slice_df = df[cols]
        for col in cols:
            tempdf = slice_df[col].map(slice_df[col].value_counts(dropna=False))
            if to_count:
                df[f'{col}_global_count'] = tempdf
                num_cols.append(f'{col}_global_count')
            if to_freq:
                df[f'{col}_global_freq'] = tempdf / tempdf.shape[0]
                num_cols.append(f'{col}_global_freq')

        self.output = [df, num_cols, cat_cols]


class EmailTransformNode(Node):
    '''
   # Transforming email domains
    '''

    def _run(self):
        emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
                  'scranton.edu': 'other', 'optonline.net': 'other',
                  'hotmail.co.uk': 'microsoft', 'comcast.net': 'other',
                  'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo',
                  'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
                  'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
                  'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
                  'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
                  'hotmail.com': 'microsoft', 'protonmail.com': 'other',
                  'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft',
                  'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',
                  'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other',
                  'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com':
                      'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att',
                  'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
                  'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo',
                  'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
                  'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft',
                  'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
                  'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
                  'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

        us_emails = ['gmail', 'net', 'edu']
        data = self.input
        for c in ['P_emaildomain', 'R_emaildomain']:
            data[c + '_bin'] = data[c].map(emails)
            data[c + '_suffix'] = data[c].map(lambda x: str(x).split('.')[-1])
            data[c + '_suffix'] = data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


class AddDeviceOSInfoNode(Node):
    def _run(self):
        data: pd.DataFrame = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        data['OS'] = np.NaN
        data['OSVersion'] = np.NaN

        def create_dev_map_dict():
            import re
            vc = data['id_30'].value_counts()
            id_30_to_OS = {}
            id_30_to_OSVersion = {}
            for s in vc.index:
                M = re.match(r'Windows\s+(\S+)', s)
                if M is not None:
                    id_30_to_OS[s] = 'Windows'
                    id_30_to_OSVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'iOS\s+(\S+)', s)
                if M is not None:
                    id_30_to_OS[s] = 'iOS'
                    id_30_to_OSVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'Mac OS X\s+(\S+)', s)
                if M is not None:
                    id_30_to_OS[s] = 'Mac'
                    id_30_to_OSVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'Android\s+(\S+)', s)
                if M is not None:
                    id_30_to_OS[s] = 'Android '
                    id_30_to_OSVersion[s] = M.groups()[0]
                    continue
                id_30_to_OS[s] = s

            return id_30_to_OS, id_30_to_OSVersion

        id_30_to_OS, id_30_to_OSVersion = create_dev_map_dict()

        found_index = data.loc[data['id_30'].isin(id_30_to_OS.keys())].index
        data.loc[found_index, 'OS'] = data.loc[found_index]['id_30'].replace(id_30_to_OS)
        data.loc[found_index, 'OSVersion'] = data.loc[found_index]['id_30'].replace(id_30_to_OSVersion)

        data['device_name'] = data['DeviceInfo'].str.split('/', expand=True)[0]
        data['device_version'] = data['DeviceInfo'].str.split('/', expand=True)[1]

        data.loc[data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        data.loc[data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        data.loc[data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        data.loc[data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        data.loc[data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        data.loc[data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        data.loc[data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        data.loc[data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        data.loc[data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        data.loc[data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        data.loc[data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        data.loc[data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        data.loc[data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        data.loc[data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        data.loc[data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        data.loc[data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        data.loc[data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
        data.loc[data.device_name.isin(
            data.device_name.value_counts()[data.device_name.value_counts() < 200].index), 'device_name'] = "Others"

        def create_browser_map_dict():
            import re
            vc = data['id_31'].value_counts()
            id_31_to_Browser = {}
            id_31_to_BrowserVersion = {}
            for s in vc.index:
                M = re.match(r'mobile safari\s+(\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'mobile safari'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'chrome\s+(\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'chrome'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'ie (\S+) for desktop', s)
                if M is not None:
                    id_31_to_Browser[s] = 'ie desktop'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'safari generic', s)
                if M is not None:
                    id_31_to_Browser[s] = 'safari'
                    id_31_to_BrowserVersion[s] = np.NaN
                    continue
                M = re.match(r'safari (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'safari'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'edge (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'edge'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'firefox (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'firefox'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'samsung browser (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'samsung browser'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'ie (\S+) for tablet', s)
                if M is not None:
                    id_31_to_Browser[s] = 'ie tablet'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'google search application (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'google search application'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'android webview (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'android webview'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'android browser (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'android browser'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'opera (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'opera'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                M = re.match(r'Generic/Android (\S+)', s)
                if M is not None:
                    id_31_to_Browser[s] = 'Generic/Android'
                    id_31_to_BrowserVersion[s] = M.groups()[0]
                    continue
                id_31_to_Browser[s] = s
                id_31_to_BrowserVersion[s] = np.NaN

            return id_31_to_Browser, id_31_to_BrowserVersion

        id_31_to_Browser, id_31_to_BrowserVersion = create_browser_map_dict()
        data['Browser'] = np.NaN
        data['BrowserVersion'] = np.NaN
        found_index = data.loc[data['id_31'].isin(id_31_to_Browser.keys())].index
        data.loc[found_index, 'Browser'] = data.loc[found_index]['id_31'].replace(id_31_to_Browser)
        data.loc[found_index, 'BrowserVersion'] = data['id_31'].replace(id_31_to_BrowserVersion)
        data.loc[
            data.Browser.isin(data.Browser.value_counts()[data.Browser.value_counts() < 10].index), 'Browser'] = "other"

        edge = {}
        edge["13"] = "2015-09-18"
        edge["14"] = "2016-02-18"
        edge["15"] = "2016-10-07"
        edge["16"] = "2017-09-26"
        edge["17"] = "2018-04-30"
        edge["18"] = "2018-11-13"
        edge_map = {}
        for k, v in edge.items():
            edge_map[str(k) + '.0'] = datetime.datetime.strptime(v, "%Y-%m-%d")

        firefox = {}
        firefox["47"] = "2016-06-07"
        firefox["48"] = "2016-08-01"
        firefox["52"] = "2017-03-07"
        firefox["55"] = "2017-08-08"
        firefox["56"] = "2017-09-28"
        firefox["57"] = "2017-11-14"
        firefox["58"] = "2018-01-23"
        firefox["59"] = "2018-03-13"
        firefox["60"] = "2018-05-09"
        firefox["61"] = "2018-06-26"
        firefox["62"] = "2018-09-05"
        firefox["63"] = "2018-10-23"
        firefox["64"] = "2018-12-11"
        firefox_map = {}
        for k, v in firefox.items():
            firefox_map[str(k) + '.0'] = datetime.datetime.strptime(v, "%Y-%m-%d")

        safari = {}
        safari["9"] = "2015-09-30"
        safari["10"] = "2016-09-20"
        safari["11"] = "2017-09-19"
        safari["12"] = "2018-09-17"
        safari_map = {}
        for k, v in safari.items():
            safari_map[str(k) + '.0'] = datetime.datetime.strptime(v, "%Y-%m-%d")

        chrome = {}
        chrome["39"] = "2014-11-18"
        chrome["43"] = "2015-05-19"
        chrome["46"] = "2015-10-13"
        chrome["49"] = "2016-03-02"
        chrome["50"] = "2016-04-13"
        chrome["51"] = "2016-05-25"
        chrome["52"] = "2016-07-20"
        chrome["53"] = "2016-08-31"
        chrome["54"] = "2016-10-12"
        chrome["55"] = "2016-12-01"
        chrome["56"] = "2017-01-25"
        chrome["57"] = "2017-03-09"
        chrome["58"] = "2017-04-19"
        chrome["59"] = "2017-06-05"
        chrome["60"] = "2017-07-25"
        chrome["61"] = "2017-09-05"
        chrome["62"] = "2017-10-17"
        chrome["63"] = "2017-12-05"
        chrome["64"] = "2018-01-24"
        chrome["65"] = "2018-03-06"
        chrome["66"] = "2018-04-17"
        chrome["67"] = "2018-05-29"
        chrome["68"] = "2018-07-24"
        chrome["69"] = "2018-09-04"
        chrome["70"] = "2018-10-16"
        chrome["71"] = "2018-12-04"
        chrome_map = {}
        for k, v in chrome.items():
            chrome_map[str(k) + '.0'] = datetime.datetime.strptime(v, "%Y-%m-%d")

        data['BrowserAge'] = np.NaN

        supported_browsers = [
            ('chrome', chrome_map),
            ('safari', safari_map),
            ('edge', edge_map),
            ('firefox', firefox_map)
        ]
        for browser, browser_map in supported_browsers:
            idx = data[data.Browser == browser][data.BrowserVersion.isin(browser_map.keys())].index
            fdata = data[data.Browser == browser]
            fdata.loc[idx, 'BrowserAge'] = ((fdata.loc[idx]['Date'].astype('datetime64[s]') - (
                fdata.loc[idx]['BrowserVersion'].replace(browser_map)).astype('datetime64[s]'))) / np.timedelta64(1,
                                                                                                                  'D')

        cat_cols.extend(['Browser', 'OS', 'OSVersion', 'device_name', 'device_version', 'BrowserVersion'])
        num_cols.extend(['BrowserAge'])


class AddCardIdNode(Node):
    def _run(self):
        data: pd.DataFrame = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        data['card_id'] = data['card1'].map(str)
        for col in ['card%d' % i for i in range(2, 6)]:
            data['card_id'] += data[col].map(lambda v: " " + str(v))
        L = LabelEncoderPopularity(convert_nan=True)
        L.fit(data['card_id'])
        data['card_id'] = L.transform(data['card_id'])
        cat_cols.append('card_id')


class AddNewCardIdNode(Node):
    def _run(self):
        data: pd.DataFrame = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]

        df = data
        # df.card_id.value_counts()

        from datetime import datetime, timedelta
        from typing import NamedTuple, List, Dict, Set
        class User(object):
            user_id: int
            card_id: str
            start_date: datetime
            last_transaction_date: datetime
            transaction_ids: List[str]

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def __repr__(self):
                return str(self.__dict__)

        users = []
        users_by_card_id_by_first_date: Dict[int, Dict[datetime, User]] = {}

        START_DATE = '1900-12-01'
        nan_startdate = datetime.strptime(START_DATE, "%Y-%m-%d")

        START_DATE = '1950-12-01'
        zero_startdate = datetime.strptime(START_DATE, "%Y-%m-%d")

        percentage = 0
        for card_id in df.card_id.value_counts().index:
            new_percentage = card_id * 100 // 19213
            if new_percentage > percentage:
                percentage = new_percentage
                if percentage % 5 == 0:
                    print(f'{percentage}%')
            user_df = df[df.card_id == card_id]
            D_cols = ['D%d' % i for i in range(1, 5)]
            filtered = user_df[['Date'] + D_cols]
            #     filtered['diff'] = filtered.Date.diff()
            for index, row in filtered.iterrows():
                dt = row['Date'].date()
                if pd.isnull(row['D1']):
                    first_transaction_date = nan_startdate
                    days_since_first_transaction = (dt - nan_startdate.date()).days
                else:
                    days_since_first_transaction = int(row['D1'])
                    if days_since_first_transaction == 0:
                        first_transaction_date = zero_startdate
                    else:
                        first_transaction_date = dt - timedelta(days=days_since_first_transaction)
                create_new = False
                if card_id not in users_by_card_id_by_first_date:
                    users_by_card_id_by_first_date[card_id] = {}
                    create_new = True
                else:
                    if first_transaction_date not in users_by_card_id_by_first_date[card_id]:
                        create_new = True
                    else:
                        u = users_by_card_id_by_first_date[card_id][first_transaction_date]
                        u.last_transaction_date = dt
                        u.transaction_ids.append(index)
                if create_new:
                    u = User(user_id=len(users),
                             card_id=card_id,
                             start_date=first_transaction_date,
                             last_transaction_date=dt,
                             transaction_ids=[index])
                    users.append(u)
                    users_by_card_id_by_first_date[card_id][first_transaction_date] = u

        l = len(users)

        index_to_new_card_id = {}
        index_to_new_start_date = {}
        percentage = 0
        for i, u in enumerate(users):
            new_percentage = i * 100 // l
            if new_percentage > percentage:
                percentage = new_percentage
                if percentage % 5 == 0:
                    print(f'{percentage}%')
            #     df_new_card_id.loc[u.transaction_ids,'new_cardId'] = u.user_id
            #     df_start_date.loc[u.transaction_ids,'start_date'] = u.start_date
            for idx in u.transaction_ids:
                index_to_new_card_id[idx] = u.user_id
                index_to_new_start_date[idx] = u.start_date

        new_card_id = np.zeros(len(df.index), dtype=np.int32)
        start_date = np.zeros(len(df.index), dtype='datetime64[ms]')
        for i, idx in enumerate(df.index):
            if idx not in index_to_new_card_id:
                print(f'Not Found {idx}')
                continue
            new_card_id[i] = index_to_new_card_id[idx]
            start_date[i] = index_to_new_start_date[idx]

        df['new_card_id'] = new_card_id
        df['start_date'] = start_date
        # df[['new_card_id', 'start_date']].to_pickle(f'{p.working_folder}/new_id.pkl')
        cat_cols.append('new_card_id')
        num_cols.append('start_date')


class AddTemporalAggregates(Node):
    params = {
        'features': [],
        'group_by': 'card_id'
    }

    def _run(self):
        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        group_by_feature = self.params['group_by']

        # Fixing same data timestamps for same card_id
        check_data = data.reset_index().set_index([group_by_feature, 'Date'])
        duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        while len(duplicate_transactions) > 0:
            print(f"Found {len(duplicate_transactions)} duplicate transactions")
            for itid, tid in enumerate(duplicate_transactions):
                print(itid)
                q = data.loc[tid]
                date = q['Date']
                card_id = q[group_by_feature]
                alldup = data[data['Date'] == date]
                alldup = alldup[alldup[group_by_feature] == card_id]
                #     print(alldup.index)
                for it, idx in enumerate(alldup.index):
                    #         print(idx)
                    data.loc[idx, 'Date'] += pd.Timedelta(seconds=it)
            check_data = data.reset_index().set_index([group_by_feature, 'Date'])
            duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        #     print(data.loc[alldup.index])
        #     break

        with mp.Pool() as Pool:

            self.output = pd.DataFrame(index=data.index)
            for nf in self.params['features']:
                if nf not in data.columns:
                    continue
                print(nf)
                df = pd.DataFrame(index=data.index)
                data_slice = data[['Date', group_by_feature, nf]].reset_index()
                args = [(data_slice, group_by_feature, nf, ws) for ws in ['1d', '2d', '3d', '7d', '30d', 5, 10, 100]]
                m = Pool.imap(aggregate_with_time_local, args)
                for i, df_agg in enumerate(m):
                    print('.')
                    assert df.shape[0] == df_agg.shape[0]
                    df = pd.concat([df, df_agg], axis=1)

                #         df = pd.concat(m, axis=1)
                #         df['TransactionID'] = data_slice['TransactionID']
                #         df.set_index('TransactionID')
                self.output = self.output.join(df)
                num_cols.extend(list(df.columns))
        #         break


class AddTransactionFrequenciesNode(Node):
    params = {
        'group_by': 'new_card_id'
    }

    def _run(self):
        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]
        group_by_feature = self.params['group_by']

        # Fixing same data timestamps for same card_id

        check_data = data.reset_index().set_index([group_by_feature, 'Date'])
        duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        while len(duplicate_transactions) > 0:
            print(f"Found {len(duplicate_transactions)} duplicate transactions")
            for itid, tid in enumerate(duplicate_transactions):
                print(itid)
                q = data.loc[tid]
                date = q['Date']
                card_id = q[group_by_feature]
                alldup = data[data['Date'] == date]
                alldup = alldup[alldup[group_by_feature] == card_id]
                #     print(alldup.index)
                for it, idx in enumerate(alldup.index):
                    #         print(idx)
                    data.loc[idx, 'Date'] += pd.Timedelta(seconds=it)
            check_data = data.reset_index().set_index([group_by_feature, 'Date'])
            duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        #     print(data.loc[alldup.index])
        #     break

        with mp.Pool() as Pool:
            self.output = pd.DataFrame(index=data.index)
            df = pd.DataFrame(index=data.index)
            data_slice = data[['Date', group_by_feature]].reset_index()

            data_slice.loc[:, 'hours'] = (data_slice.Date - data_slice.Date.iloc[0]).dt.total_seconds() / 3600

            args = [(data_slice, group_by_feature, ws) for ws in ['1d', '2d', '3d', '7d', '30d', 5, 10, 100]]
            m = Pool.imap(aggregate_transaction_frequencies, args)
            for i, df_agg in enumerate(m):
                print('.')
                assert df.shape[0] == df_agg.shape[0]
                df = pd.concat([df, df_agg], axis=1)

            #         df = pd.concat(m, axis=1)
            #         df['TransactionID'] = data_slice['TransactionID']
            #         df.set_index('TransactionID')
            self.output = self.output.join(df)
            num_cols.extend(list(df.columns))
    #         break


class CorrectScreenWidthHeightTypeNode(Node):
    def _run(self):
        data = self.input
        data['screen_width'] = data['screen_width'].fillna(np.NaN).astype(np.float16)
        data['screen_height'] = data['screen_height'].replace('None', np.NaN).astype(np.float16)


class FindUselessForTrainingFeaturesNode(Node):
    params = {
        'threshold': 0.2,
        'features': []
    }

    def _run(self):
        from sklearn.preprocessing import StandardScaler
        data = self.input
        train_ids = data[data.isFraud >= 0].index
        test_ids = data[data.isFraud < 0].index
        scaler = StandardScaler()
        feature_skew = {}
        for col in self.params['features']:

            if col in feature_skew:
                continue
            data_col = data[col].replace(np.inf, np.NaN).dropna()
            normalized_data = pd.DataFrame(index=data_col.index)
            try:
                normalized_data['data'] = scaler.fit_transform(data_col.values.reshape((-1, 1)))
            except ValueError as ex:
                print(f'Error in {col}')
                feature_skew[col] = None
                continue
            train_col = normalized_data['data'].reindex(train_ids).dropna()
            test_col = normalized_data['data'].reindex(test_ids).dropna()

            train_avg = train_col.mean()
            test_avg = test_col.mean()
            print(col, abs(test_avg))
            feature_skew[col] = abs(test_avg)
        bad_for_training_features = []
        for k, v in feature_skew.items():
            if v is None or np.isnan(v) or v > self.params['threshold']:
                bad_for_training_features.append(k)

        self.output = bad_for_training_features


class FindFalsePredictionsNode(Node):
    params = {
        'cutoff': 0.3
    }

    def _run(self):
        data = self.input[0][['isFraud']]
        train = data[data.isFraud >= 0]

        oof = self.input[1].set_index('TransactionID')['isFraud']
        train['pred'] = oof
        cutoff = 0.3
        true_positives = train[train.pred_0 >= cutoff][train.isFraud == 1]
        false_positives = train[train.pred_0 >= cutoff][train.isFraud == 0]
        false_negatives = train[train.pred_0 < cutoff][train.isFraud == 1]
        true_negatives = train[train.pred_0 < cutoff][train.isFraud == 0]
        print(f'TP: {len(true_positives)}, FP: {len(false_positives)}')
        print(f'FN: {len(false_negatives)}, TN: {len(true_negatives)}')
        self.output = [false_positives, false_negatives]
