import pandas as pd


def aggregate_with_time_local(args):
    data_slice, group_column, num_feature, window_size, time_feature = args
    #     data_slice = data[['Date',group_column,num_feature]].reset_index()
    gb = data_slice.groupby([group_column])
    q = gb.rolling(window_size, on=time_feature, min_periods=1)[num_feature].agg(['mean', 'std'])
    ds = data_slice.set_index([group_column, time_feature])
    ds = ds.join(q)
    ds = ds.set_index('TransactionID').sort_index()
    to_mean = ds[num_feature] / ds['mean']
    to_std = ds[num_feature] / ds['std']
    df = pd.DataFrame()
    df[f'{num_feature}_by_{group_column}_ws{window_size}_to_mean'] = to_mean
    df[f'{num_feature}_by_{group_column}_ws{window_size}_to_std'] = to_std
    return df


def transaction_velocity_agg(x):
    return (x - x.iloc[0]).mean()


def aggregate_transaction_frequencies(args):
    data_slice, group_column, window_size, time_feature = args

    ds = data_slice.set_index([group_column, time_feature])

    gb = data_slice.groupby([group_column])
    q = gb.rolling(window_size, on=time_feature, min_periods=1, center=False)['hours'].agg(transaction_velocity_agg)
    ds[f'Transaction_freq_{window_size}_past'] = q

    if isinstance(window_size, int):
        qc = gb.rolling(window_size, on='Date', min_periods=1, center=True)['hours'].agg(transaction_velocity_agg)
        ds[f'Transaction_freq_{window_size}_center'] = qc

    ds = ds.drop(['hours'], axis=1)

    ds = ds.set_index('TransactionID').sort_index()
    return ds
