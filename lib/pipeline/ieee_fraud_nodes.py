from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from lib.pipeline.node import Node
from lib.pipeline.pipeline import *
from lib.io import *
import os
from sklearn import preprocessing


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
        dataset["Date"] = dataset['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        dataset['_Weekdays'] = dataset['Date'].dt.dayofweek
        dataset['_Hours'] = dataset['Date'].dt.hour
        dataset['_Days'] = dataset['Date'].dt.day

        # Taken from Anna Notebook
        dataset['isNight'] = dataset['Hours'].map(lambda x: 1 if (x >= 23 or x < 5) else 0)
        dataset['DT_M'] = (dataset['Date'].dt.year - 2017) * 12 + dataset['Date'].dt.month
        dataset['DT_W'] = (dataset['Date'].dt.year - 2017) * 52 + dataset['Date'].dt.weekofyear
        dataset['DT_D'] = (dataset['Date'].dt.year - 2017) * 365 + dataset['Date'].dt.dayofyear

        dataset.drop(['TransactionDT'], axis=1, inplace=True)


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

        data['card1_count_full'] = data['card1'].map(data.value_counts(dropna=False))

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

        for c in ['P_emaildomain', 'R_emaildomain']:
            train[c + '_bin'] = train[c].map(emails)
            train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
            train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')