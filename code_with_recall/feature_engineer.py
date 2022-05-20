import datetime
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc


def customer_feature_engineer(samples, data, data_lw, article_w2v_df):

    df = data[data.customer_id.isin(samples.customer_id.unique())]

    # w2v特征用户侧聚合

    df = df.merge(article_w2v_df, on='article_id', how='left')

    w2v_feats = [c for c in article_w2v_df.columns if c not in ['article_id']]

    for f in w2v_feats:
        df[f] = df[f].fillna(0)

    tmp = df.groupby('customer_id')[w2v_feats].agg('mean').reset_index()

    tmp.columns = ['customer_id'] + ['customer_{}_mean'.format(c) for c in w2v_feats]

    print(tmp)

    samples = samples.merge(tmp, on='customer_id', how='left')

    df.drop(columns=w2v_feats)

    gc.collect()


    group = df.groupby('customer_id')
    # group = data[data.customer_id.isin(samples.customer_id.unique())].groupby('customer_id')

    tmp = group.agg({
        'article_id': 'count',
        'price': lambda x: sum(np.array(x) > x.mean()),
        'sales_channel_id': lambda x: sum(x == 2),
    }).rename(columns={
        'article_id': 'n_purchase',
        'price': 'n_transactions_bigger_mean',
        'sales_channel_id': 'n_online_articles'
    }).reset_index()

    samples = samples.merge(tmp, on='customer_id', how='left')

    # agg_cols = ['min', 'max', 'mean', 'std']

    agg_cols = ['min', 'max', 'mean', 'std', 'median', 'sum']

    tmp = group['price'].agg(agg_cols).reset_index()

    tmp.columns = ['customer_id'] + ['customer_price_{}'.format(col) for col in agg_cols]

    tmp['customer_price_max_minus_min'] = tmp['customer_price_max'] - tmp['customer_price_min']

    samples = samples.merge(tmp, on='customer_id', how='left')

    tmp = group.agg({
        'article_id': 'nunique',
        'sales_channel_id': lambda x: sum(x == 1),
    }).rename(columns={
        'article_id': 'n_purchase_nuniq',
        'sales_channel_id': 'n_store_articles'
    }).reset_index()

    samples = samples.merge(tmp, on='customer_id', how='left')

    '''
    # 购买时间间隔
    df['purchase_interval_time'] = (df['t_dat'] - df.groupby('customer_id')['t_dat'].shift()).dt.days
    agg_cols = ['min', 'max', 'mean', 'std', 'median']
    tmp = df.groupby('customer_id')['purchase_interval_time'].agg(agg_cols).reset_index()
    tmp.columns = ['customer_id'] + ['purchase_interval_time_{}'.format(c) for c in agg_cols]
    samples = samples.merge(tmp, on='customer_id', how='left')
    '''

    tmp = group.agg({
        'sales_channel_id': 'mean',
    }).rename(columns={
        'sales_channel_id': 'customer_sales_channel_mean'
    }).reset_index()
    samples = samples.merge(tmp, on='customer_id', how='left')

    del tmp

    gc.collect()

    return samples

def article_feature_engineer(samples, data, data_last_week, time_max):
    tmp = data.groupby('article_id')['customer_id'].agg('count').reset_index()

    tmp.columns = ['article_id', 'purchased_cnt_global_hist']

    samples = samples.merge(tmp, on='article_id', how='left')

    agg_cols = ['min', 'max', 'mean', 'std']

    tmp = data.groupby('article_id')['price'].agg(agg_cols).reset_index()

    tmp.columns = ['article_id'] + ['article_price_{}'.format(col) for col in agg_cols]

    samples = samples.merge(tmp, on='article_id', how='left')

    # article 时间特征

    # cols = ['mean', 'sum', 'min', 'max', 'std']
    cols = ['mean', 'min', 'max', 'std']

    data['pop_factor'] = (time_max - data['t_dat']).dt.days

    # print(data.pop_factor.describe())

    tmp = data.groupby(['article_id'])['pop_factor'].agg(cols).reset_index()

    tmp.columns = ['article_id'] + ['article_time_{}'.format(col) for col in cols]

    samples = samples.merge(tmp, on='article_id', how='left')

    del data['pop_factor'], tmp

    gc.collect()

    tmp = data.groupby('article_id')['sales_channel_id'].agg('mean').reset_index()
    tmp.columns = ['article_id', 'article_sales_channel_mean']
    samples = samples.merge(tmp, on='article_id', how='left')

    del tmp

    gc.collect()

    return samples

def interact_feature_engineer(samples, data, date_ths, data_lm, data_lw, articles, customers):
    # cols = [f for f in articles.columns if f not in ['article_id']]
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']

    new_cols = []

    tcols = ['product_type_no', 'product_group_name', 'colour_group_code', 'index_group_no', 'department_no', 'garment_group_no']

    data_ = data[data.customer_id.isin(samples.customer_id.unique())].merge(articles, on='article_id', how='left')

    for col in tqdm(cols):
        tmp = data_.groupby(['customer_id', col])['article_id'].agg('count').reset_index()
        new_col = 'customer_{}_hist_cnt'.format(col)
        tmp.columns = ['customer_id', col, new_col]
        new_cols += [new_col]

        # tmp = data_.groupby(['customer_id', col])['t_dat'].agg(['count', 'max']).reset_index()
        # tmp.columns = ['customer_id', col, 'customer_{}_hist_cnt'.format(col), 'purchase_corr_{}_max_time'.format(col)]
        # tmp['purchase_corr_{}_max_time'.format(col)] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['purchase_corr_{}_max_time'.format(col)]).dt.days

        #  new_cols += ['customer_{}_hist_cnt'.format(col), 'purchase_corr_{}_max_time'.format(col)]
        # new_cols += ['customer_{}_hist_cnt'.format(col)]

        samples = samples.merge(tmp, on=['customer_id', col], how='left')

        # 用户购买最多的类型以及次数
        # if col in tcols:
        # tmp = tmp.sort_values(new_col, ascending=False).groupby('customer_id').head(1).reset_index(drop=True)
        # tmp.columns = ['customer_id', 'purchase_most_times_{}'.format(col), 'purchase_most_times_{}_cnt'.format(col)]
        # samples = samples.merge(tmp, on=['customer_id'], how='left')


    # 上次购买候选物品距今时间

    tmp = data.groupby(['customer_id', 'article_id'])['t_dat'].agg('max').reset_index()

    tmp['purchase_corr_article_max_time'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days

    new_cols += ['purchase_corr_article_max_time']

    samples = samples.merge(tmp[['customer_id', 'article_id', 'purchase_corr_article_max_time']],
                            on=['customer_id', 'article_id'], how='left')

    '''
    # 过去购买过该物品次数统计
    tmp = data.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
    tmp.columns = ['customer_id', 'article_id', 'purchase_corr_article_cnt']
    new_cols += ['purchase_corr_article_cnt']
    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')
    '''

    # 上周同属性人群购买候选物品次数
    cols = ['age_group', 'postal_code']
    data_lw = data_lw.merge(customers[['customer_id'] + cols], on='customer_id', how='left')
    for col in cols:
        tmp = data_lw.groupby([col, 'article_id'])['t_dat'].agg('count').reset_index()
        tmp.columns = [col, 'article_id', 'same_{}_purchase_corr_article_cnt_lw'.format(col)]
        new_cols += ['same_{}_purchase_corr_article_cnt_lw'.format(col)]
        samples = samples.merge(tmp, on=[col, 'article_id'], how='left')

        # agg_cols = ['min', 'max', 'mean']
        # tmp = data_lw.groupby([col, 'article_id'])['price'].agg(agg_cols).reset_index()
        # tmp.columns = [col, 'article_id'] + ['same_{}_purchase_corr_article_price_{}'.format(col, agg_col) for agg_col in agg_cols]
        # new_cols += ['same_{}_purchase_corr_article_price_{}'.format(col, agg_col) for agg_col in agg_cols]
        # samples = samples.merge(tmp, on=[col, 'article_id'], how='left')


    cols = ['count']

    # 过去三天购买过的物品次数统计
    last_3days = 3 # 30

    last_3days_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_3days)

    tmp = data_lw[data_lw.t_dat >= last_3days_date].groupby(['customer_id', 'article_id'])['price'].agg(
        cols).reset_index()

    new_col = ['customer_article_last_3days_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')


    # 过去两周购买过的物品次数统计
    last_2weeks = 14

    last_2weeks_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_2weeks)

    tmp = data_lm[data_lm.t_dat >= last_2weeks_date].groupby(['customer_id', 'article_id'])['price'].agg(
        cols).reset_index()

    new_col = ['customer_article_last_2weeks_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')

    '''
    # 过去一个月购买过的物品次数统计

    cols = ['count']

    tmp = data_lm.groupby(['customer_id', 'article_id'])['price'].agg(cols).reset_index()

    new_col = ['customer_article_last_month_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')
    '''
    # 过去一周购买过的物品次数统计

    tmp = data_lw.groupby(['customer_id', 'article_id'])['price'].agg(cols).reset_index()

    new_col = ['customer_article_last_week_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')

    # 过去一天购买过的物品次数统计

    tmp = data_lw[data_lw.t_dat == data_lw.t_dat.max()].groupby(['customer_id', 'article_id'])['price'].agg(
        cols).reset_index()

    new_col = ['customer_article_last_day_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')


    # 历史最近一次点击距今时间

    tmp = data_.groupby('customer_id')['t_dat'].agg('max').reset_index()

    # tmp['latest_purchase_time_sub'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - pd.to_datetime(tmp['t_dat'])).dt.days

    tmp['latest_purchase_time_sub'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days

    samples = samples.merge(tmp[['customer_id', 'latest_purchase_time_sub']], on='customer_id', how='left')

    new_cols.append('latest_purchase_time_sub')

    del data_, tmp

    gc.collect()


    # samples['latest_purchase_time_sub_purchase_corr_article_max_time'] = samples['latest_purchase_time_sub'] - samples['purchase_corr_article_max_time']
    # new_cols.append('latest_purchase_time_sub_purchase_corr_article_max_time')

    '''
    # 时间特征
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
    for col in tqdm(cols):

        samples['latest_purchase_time_sub_purchase_corr_{}_max_time'.format(col)] = samples['latest_purchase_time_sub'] - samples['purchase_corr_{}_max_time'.format(col)]

        samples.drop(columns=['purchase_corr_{}_max_time'.format(col)], inplace=True)
    '''
    '''
    # 类别流行度特征
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
    data_lw_ = data_lw.merge(articles[['article_id'] + cols], on='article_id', how='left')

    for col in tqdm(cols):
        tmp = data_lw_.groupby(col)['customer_id'].agg('count').reset_index()

        tmp.columns = [col, '{}_popularity_last_week'.format(col)]

        samples = samples.merge(tmp, on=col, how='left')

        samples['popularity_last_week_article_div_{}'.format(col)] = samples['popularity'] / samples['{}_popularity_last_week'.format(col)]

        new_cols += ['{}_popularity_last_week'.format(col), 'popularity_last_week_article_div_{}'.format(col)]

        new_cols += ['{}_popularity_last_week'.format(col)]

    del data_lw_, tmp

    gc.collect()
    '''

    samples = reduce_mem(samples, new_cols)

    return samples

def reduce_mem(df, cols):
    df = df.copy()
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    gc.collect()

    return df

def feature_engineer(data, date, samples, customers, articles, article_w2v_df, last_days=7, dtype='train'):

    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    pop_begin_date = str(pop_begin_date).split(' ')[0]

    print(pop_begin_date, begin_date, date)

    data_lw = data[(data.t_dat >= pop_begin_date) & (data.t_dat <= begin_date)]
    data_hist = data[data.t_dat <= begin_date]

    last_month_days = 30
    last_month_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)
    last_month_date = str(last_month_date).split(' ')[0]
    data_lm = data_hist[data_hist.t_dat >= last_month_date]

    # 特征工程
    print('customer feature engineer')
    samples = customer_feature_engineer(samples, data_hist, data_lw, article_w2v_df)

    print('article feature engineer')
    samples = article_feature_engineer(samples, data_hist, data_lw, datetime.datetime.strptime(begin_date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    # samples = samples.merge(articles, on='article_id', how='left')
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']
    samples = samples.merge(articles[['article_id'] + cols], on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    tfidf_df = pd.read_csv('tfidf_feats/offline_{}.csv'.format(dtype)).drop(columns=['type'])
    samples = samples.merge(tfidf_df, on='customer_id', how='left')

    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    print('interact feature engineer')
    samples = interact_feature_engineer(samples, data_hist, begin_date, data_lm,
                                        data_lw, articles, customers)

    del data_hist, data_lw, data_lm

    gc.collect()

    return samples