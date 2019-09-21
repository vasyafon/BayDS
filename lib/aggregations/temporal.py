import pandas as pd


def aggregate_with_time_local(args):
    data_slice, group_column, num_feature, window_size = args
    #     data_slice = data[['Date',group_column,num_feature]].reset_index()
    gb = data_slice.groupby([group_column])
    q = gb.rolling(window_size, on='Date', min_periods=1)[num_feature].agg(['mean', 'std'])
    ds = data_slice.set_index([group_column, 'Date'])
    ds = ds.join(q)
    ds = ds.set_index('TransactionID').sort_index()
    to_mean = ds[num_feature] / ds['mean']
    to_std = ds[num_feature] / ds['std']
    df = pd.DataFrame()
    df[f'{num_feature}_by_{group_column}_ws{window_size}_to_mean'] = to_mean
    df[f'{num_feature}_by_{group_column}_ws{window_size}_to_std'] = to_std
    return df

