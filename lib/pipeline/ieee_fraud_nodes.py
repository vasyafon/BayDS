from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from .node import Node
from .pipeline import *
from ..io import *
import numpy as np
import os
from sklearn import preprocessing
from ..encoding import LabelEncoderPopularity
from ..aggregations.temporal import aggregate_with_time_local
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
        START_DATE = '2017-12-01'
        startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

        dataset = self.input
        dataset["Date"] = dataset['Transaction  DT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        dataset['Weekdays'] = dataset['Date'].dt.dayofweek
        dataset['Hours'] = dataset['Date'].dt.hour
        dataset['Days'] = dataset['Date'].dt.day

        # Taken from Anna Notebook
        dataset['isNight'] = dataset['Hours'].map(lambda x: 1 if (x >= 23 or x < 5) else 0)
        dataset['DT_M'] = (dataset['Date'].dt.year - 2017) * 12 + dataset['Date'].dt.month
        dataset['DT_W'] = (dataset['Date'].dt.year - 2017) * 52 + dataset['Date'].dt.weekofyear
        dataset['DT_D'] = (dataset['Date'].dt.year - 2017) * 365 + dataset['Date'].dt.dayofyear

        # dataset.drop(['TransactionDT'], axis=1, inplace=True)


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

            dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype(str)
            dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype(str)

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

        cat_cols.extend(['Browser', 'OS', 'OSVersion', 'device_name', 'device_version'])
        num_cols.extend(['BrowserAge', 'BrowserVersion', 'screen_height', 'screen_width'])


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


class AddTemporalAggregates(Node):
    params = {
        'features': [],
        'group_by': 'card_id'
    }

    def _run(self):
        with mp.Pool() as Pool:
            data = self.input[0]
            num_cols = self.input[1]
            cat_cols = self.input[2]
            group_by_feature = self.params['group_by']
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
                    df = pd.concat([df, df_agg], axis=1)
                #         df = pd.concat(m, axis=1)
                #         df['TransactionID'] = data_slice['TransactionID']
                #         df.set_index('TransactionID')
                self.output = self.output.join(df)
                num_cols.extend(list(df.columns))
        #         break
