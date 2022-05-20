import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')


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

from time import time

import datetime


def process(customers, articles):
    cols = ['club_member_status', 'fashion_news_frequency', 'postal_code']

    for col in tqdm(cols):
        dic = dict(zip(customers[col].unique(), range(customers[col].nunique())))

        customers[col] = customers[col].map(dic)

    cols = [col for col in articles.columns if articles[col].dtype is articles['article_id'].dtype][1:-1] + ['detail_desc']

    dic_dict = {}

    for col in tqdm(cols):
        dic = dict(zip(articles[col].unique(), range(articles[col].nunique())))

        articles[col] = articles[col].map(dic)

        dic_dict[col] = dic

    # articles.drop(columns='detail_desc', inplace=True)

    '''
    tmp = transactions_train.merge(articles[['article_id', 'index_group_no']], on='article_id', how='left').groupby(['customer_id', 'index_group_no'])['article_id'].agg('count').reset_index()
    tmp.sort_values('article_id', ascending=False, inplace=True)
    tmp = tmp.groupby('customer_id').head(1).reset_index(drop=True)
    tmp.columns = ['customer_id', 'attribute', 'cnt']

    customers = customers.merge(tmp[['customer_id', 'attribute']], on='customer_id', how='left')
    '''

    customers['age_group'] = 0
    customers.loc[customers.age < 20, 'age_group'] = 1
    customers.loc[(customers.age >= 20) & (customers.age < 30), 'age_group'] = 2
    customers.loc[(customers.age >= 30) & (customers.age < 40), 'age_group'] = 3
    customers.loc[(customers.age >= 40) & (customers.age < 50), 'age_group'] = 4
    customers.loc[(customers.age >= 50) & (customers.age < 60), 'age_group'] = 5
    customers.loc[(customers.age >= 60) & (customers.age < 70), 'age_group'] = 6
    customers.loc[(customers.age >= 70) & (customers.age < 80), 'age_group'] = 7
    customers.loc[(customers.age >= 80), 'age_group'] = 8

    return customers, articles, dic_dict


def construct_samples_valid(data, dates):
    samples = []
    for date in tqdm(dates):
        # df_sample, df_target = data[data.t_dat <= date], data[data.t_dat > date]
        # df_sample, df_target = data[(data.t_dat <= date) & (data.t_dat >= '2020-08-15')], data[data.t_dat > date]
        df_sample, df_target = data[(data.t_dat <= date)], data[data.t_dat > date]
        df_sample = df_sample[df_sample.customer_id.isin(df_target.customer_id.unique())]

        df = df_target.groupby('customer_id')['article_id'].agg(lambda x: list(set(x))).reset_index()
        df.columns = ['customer_id', 'label']

        tmp = df_sample.groupby('customer_id')['article_id'].agg(list).reset_index()
        df = df.merge(tmp, on='customer_id', how='left')
        tmp = df_sample.groupby('customer_id')['price'].agg(list).reset_index()
        df = df.merge(tmp, on='customer_id', how='left')

        tmp = df_sample.groupby('customer_id')['t_dat'].agg(list).reset_index()
        df = df.merge(tmp, on='customer_id', how='left')

        samples.append(df)
    del df_sample, df_target
    gc.collect()

    samples = pd.concat(samples)

    return samples


import datetime


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



from collections import defaultdict

import math

'''
def customer_feature_engineer(samples, data, data_lw):
    tmp = data.groupby('customer_id')['article_id'].agg('count').reset_index()

    tmp.columns = ['customer_id', 'puchase_cnt_global_hist']

    samples = samples.merge(tmp, on='customer_id', how='left')


    agg_cols = ['min', 'max', 'mean', 'std']

    tmp = data.groupby('customer_id')['price'].agg(agg_cols).reset_index()

    tmp.columns = ['customer_id'] + ['customer_price_{}'.format(col) for col in agg_cols]

    samples = samples.merge(tmp, on='customer_id', how='left')


    del tmp

    gc.collect()

    return samples
'''

def customer_feature_engineer(samples, data, data_lw):

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

def interact_feature_engineer(samples, data, date_ths, data_lm, data_lw):
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


def construct_samples(data, date, last_days=7, recall_num=100, dtype='train'):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    pop_begin_date = str(pop_begin_date).split(' ')[0]

    print(pop_begin_date, begin_date, date)

    # 热门项目召回列表
    data_lw = data[(data.t_dat >= pop_begin_date) & (data.t_dat <= begin_date)]

    dummy_dict = data_lw['article_id'].value_counts()

    dummy_list = [(k, v) for k, v in dummy_dict.items()][:recall_num]

    target_df = data[(data.t_dat <= date) & (data.t_dat > begin_date)]

    target = target_df.groupby('customer_id')['article_id'].agg(list).reset_index()

    target.columns = ['customer_id', 'label']

    data_hist = data[data.t_dat <= begin_date]

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    data_lm = data_hist[data_hist.t_dat >= last_month_date]

    tmp = data_hist.groupby('customer_id')['t_dat'].agg('max').reset_index()

    target = target.merge(tmp, on='customer_id', how='left')

    print(last_month_date)

    target = target[target.t_dat >= last_month_date]

    del target['t_dat']

    gc.collect()

    samples = []

    hit = 0

    for cust, labels in tqdm(target.values):

        h = 0

        for cart, pv in dummy_list:

            if cart in labels:

                sample = [cust, cart, 1]

                h += 1

            else:

                sample = [cust, cart, 0]

            samples.append(sample)

        hit += h / len(labels)

    print('HIT: ', hit / len(target))

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'label'])

    # 召回过去一个月购买过的项目
    # samples_pd = data_lm[data_lm.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]
    purchase_df = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())].groupby('customer_id').tail(100).reset_index(drop=True)
    # purchase_df = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())]
    samples_pd = purchase_df[['customer_id', 'article_id']]

    pd_df = samples_pd.groupby('customer_id')['article_id'].agg(list).reset_index()

    samples_pd = []

    pd_df = pd_df.merge(target, on='customer_id', how='left')

    for cust, carts, label in pd_df.values:

        for cart in carts:

            if cart in label:

                samples_pd.append([cust, cart, 1])

            else:

                samples_pd.append([cust, cart, 0])

    samples_pd = pd.DataFrame(samples_pd, columns=['customer_id', 'article_id', 'label'])

    # samples_pd['popularity'] = samples_pd['article_id'].map(dummy_dict)

    print(samples_pd.shape)

    # print(samples_pd.popularity.value_counts())


    samples = pd.concat([samples, samples_pd[list(samples.columns)]])
    print(samples.shape)
    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
    print(samples.shape)
    print(samples.label.mean())

    # BinaryNet进行召回
    binary_samples = pd.read_csv('binary_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    # if dtype == 'valid':
    binary_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)

    samples = samples.merge(binary_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('BinaryNet召回后: ', samples.label.mean())
    del binary_samples
    gc.collect()

    # ItemCF进行召回
    itemcf_samples = pd.read_csv('itemcf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    # if dtype == 'valid':
    itemcf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)

    samples = samples.merge(itemcf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('ItemCF召回后: ', samples.label.mean())
    del itemcf_samples
    gc.collect()

    # UserCF进行召回
    usercf_samples = pd.read_csv('usercf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    usercf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(usercf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('UserCF召回后: ', samples.label.mean())
    del usercf_samples
    gc.collect()

    # cate pop召回
    cp_samples = pd.read_csv('cate_pop_recall/{}.csv'.format(dtype))
    cp_samples['customer_id'] = cp_samples['customer_id'].map(cust_ex_dic)
    cp_samples['article_id'] = cp_samples['article_id'].map(art_ex_dic)
    cp_samples = cp_samples[cp_samples.customer_id.isin(samples.customer_id.unique())]
    samples = samples.merge(cp_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    # samples = samples.merge(cp_samples[['customer_id', 'article_id', 'label']],
    #                         on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('Cate_Pop召回后: ', samples.label.mean())

    samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))

    tmp = samples.groupby('customer_id')['label'].agg('sum').reset_index()
    print('召回后具有正样本的用户占比: ', len(tmp[tmp.label > 0]) / len(tmp))

    # 特征工程

    samples = customer_feature_engineer(samples, data_hist, data_lw)

    samples = article_feature_engineer(samples, data_hist, data_lw, datetime.datetime.strptime(begin_date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    # samples = samples.merge(articles, on='article_id', how='left')
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']
    samples = samples.merge(articles[['article_id'] + cols], on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')
    '''
    samples = samples.merge(pt_w2v_df, on='product_type_no', how='left')
    # samples = samples.merge(pg_w2v_df, on='product_group_name', how='left')

    for col, df in df_dict.items():

        samples = samples.merge(df, on=col, how='left')
    '''


    tfidf_df = pd.read_csv('tfidf_feats/offline_{}.csv'.format(dtype)).drop(columns=['type'])
    samples = samples.merge(tfidf_df, on='customer_id', how='left')


    # cols = ['product_type_no', 'department_no']
    # for col in tqdm(cols):
    #     tfidf_df = pd.read_csv('tfidf_feats/{}/offline_{}.csv'.format(col, dtype))
    #     samples = samples.merge(tfidf_df, on='customer_id', how='left')
    '''
    tmp = data_lw.groupby(['article_id', 't_dat'])['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 't_dat', 'cnt']
    tmp = tmp.set_index(['article_id', 't_dat'])['cnt'].unstack().reset_index()
    tmp.columns = ['article_id'] + ['last_day{}_cnt'.format(c) for c in range(1, len(tmp.columns))]
    samples = samples.merge(tmp, on='article_id', how='left')
    del tmp    
    gc.collect()
    '''

    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    samples = interact_feature_engineer(samples, data_hist, begin_date, data_lm,
                                        data_lw)

    del data_hist, data_lw, target, data_lm

    gc.collect()

    return samples


import lightgbm as lgb


def ranker(train, valid):
    train.sort_values(by=['customer_id'], inplace=True)

    g_train = train.groupby(['customer_id'], as_index=False).count()["label"].values

    valid.sort_values(by=['customer_id'], inplace=True)

    g_val = valid.groupby(['customer_id'], as_index=False).count()["label"].values

    del_cols = []  # + list(tfidf_df.columns)[:5]# ['popularity', 'popularity_last_1days', 'popularity_last_month']

    feats = [f for f in train.columns if f not in ['customer_id', 'article_id', 'label', 'prob'] + del_cols]

    # cate_feats = [c for c in list(customers.columns) + list(articles.columns) if c not in ['customer_id', 'article_id']]
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1)

    lgb_ranker.fit(train[feats], train['label'], group=g_train,
                   eval_set=[(valid[feats], valid['label'])],
                   eval_group=[g_val], eval_at=[12], eval_metric=['map', ],
                   early_stopping_rounds=100, # 50,
                   verbose=50,
                   # categorical_feature=cate_feats
                   )

    print(lgb_ranker.best_score_)

    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = lgb_ranker.feature_importances_

    print(importance_df.sort_values('importance', ascending=False).head(20))

    valid['prob'] = lgb_ranker.predict(valid[feats], num_iteration=lgb_ranker.best_iteration_)

    return lgb_ranker, valid

def get_customer_group(valid, begin_day):
    cnt1, cnt2, cnt3 = 0, 0, 0

    cg1, cg2, cg3 = [], [], []

    for cust_id, labels, hist_arts, hist_dates, prices in tqdm(
            valid[['customer_id', 'label', 'article_id', 't_dat', 'price']].values):

        if hist_arts is np.nan:

            cnt3 += 1

            cg3.append(cust_id)

        elif hist_dates[-1] >= begin_day:

            cnt1 += 1

            cg1.append(cust_id)

        else:

            cnt2 += 1

            cg2.append(cust_id)

    print(cnt1, cnt2, cnt3)

    return cg1, cg2, cg3


def MAP(pred, label):
    n = min(len(label), 12)

    res = 0

    for k in range(n):

        if pred[k] in label:

            p = 0

            # for pa in pred[:k + 1]:

            for i, pa in enumerate(pred[:k + 1]):

                if pa in label and pa not in pred[:i]:
                    p += 1

            p /= (k + 1)

            res += p / n

    return res


def prediction(valid, dummy_list=None):
    un, ap = 0, 0

    for cust_id, labels, rec in tqdm(valid[['customer_id', 'label', 'rec_list']].values):
        # rec = rec[:12] if len(rec) >= 12 else rec + dummy_list[:12 - len(rec)]
        rec = rec[:12]

        un += 1

        ap += MAP(rec, labels)

    map12 = ap / un

    print('[CG1] MAP@12: ', map12)

cust_dic = np.load('map_dict/cust.npy', allow_pickle=True).item()
cust_ex_dic = {}
for k, v in cust_dic.items():
    cust_ex_dic[v] = k
art_dic = np.load('map_dict/art.npy', allow_pickle=True).item()
art_ex_dic = {}
for k, v in art_dic.items():
    art_ex_dic[v] = k

articles = pd.read_csv('articles.csv', dtype={'article_id': str})
customers = pd.read_csv('customers.csv')
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
samp_sub = pd.read_csv('sample_submission.csv')
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

customers, articles, dic_dict = process(customers, articles)

last_days, size = 90, 20 # 90, 20

# article_w2v_df = generate_w2v_feats(last_days=last_days, size=size)
article_w2v_df = pd.read_csv('w2v_feats/offline_w2v_feats.csv', dtype={'article_id': str})
'''
pt_w2v_df = pd.read_csv('w2v_feats/product_type_no/offline_w2v_feats_size10.csv')
if 'product_type_no' in dic_dict:
    pt_w2v_df['product_type_no'] = pt_w2v_df['product_type_no'].map(dic_dict['product_type_no'])

pg_w2v_df = pd.read_csv('w2v_feats/product_group_name/offline_w2v_feats_size10.csv')
if 'product_group_name' in dic_dict:
    pg_w2v_df['product_group_name'] = pg_w2v_df['product_group_name'].map(dic_dict['product_group_name'])

# cols = ['graphical_appearance_no', 'colour_group_code', 'department_no', 'index_code', 'section_no', 'garment_group_no']
cols = ['department_no', 'index_code', 'section_no'] # 0.04448
# cols = ['department_no', 'index_code'] # 0.04396
# cols = ['section_no'] # 0.04379
# cols = ['department_no', 'index_code', 'section_no', 'garment_group_no'] # 0.04409534450146968
df_dict = {}
for col in cols:
    tmp = pd.read_csv('w2v_feats/{}/offline_w2v_feats_size10_ld90.csv'.format(col))
    if col in dic_dict:
        tmp[col] = tmp[col].map(dic_dict[col])
    df_dict[col] = tmp
'''

recall_num = 20  # 30 # 100

train = construct_samples(transactions_train, '2020-09-15', last_days=7, recall_num=recall_num, dtype='train')

valid = construct_samples(transactions_train, '2020-09-22', last_days=7, recall_num=recall_num, dtype='valid')

print(train.shape, valid.shape)

model, valid = ranker(train, valid)

dates = ['2020-09-15']

valid_df = construct_samples_valid(transactions_train, dates)

last_days = 30

begin_day = datetime.datetime.strptime(dates[0], '%Y-%m-%d') - datetime.timedelta(days=last_days)

cg1, cg2, cg3 = get_customer_group(valid_df, begin_day)

valid_cg1 = valid_df[valid_df.customer_id.isin(cg1)]

tmp = valid.sort_values('prob', ascending=False).groupby('customer_id')['article_id'].agg(list).reset_index()
tmp.columns = ['customer_id', 'rec_list']
res = valid_cg1.merge(tmp, on='customer_id', how='left')

prediction(res)

cust_dic = np.load('map_dict/cust.npy', allow_pickle=True).item()
art_dic = np.load('map_dict/art.npy', allow_pickle=True).item()

save_path = 'offline_prob/cg1_model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

valid['customer_id'] = valid['customer_id'].map(cust_dic)
valid['article_id'] = valid['article_id'].map(art_dic)
prob_df = valid[['customer_id', 'article_id', 'prob']]
prob_df = reduce_mem(prob_df, cols=['customer_id', 'article_id', 'prob'])
prob_df.to_csv(save_path + 'prob.csv', index=False)