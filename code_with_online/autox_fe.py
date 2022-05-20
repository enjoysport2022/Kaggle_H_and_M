import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime


def interact_feature_engineer(samples, data, customers, articles, uid, iid, time_col):
    date_ths = str(data[time_col].max())[:10]

    last_month = 30
    last_month_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_month)
    data_lm = data[data[time_col] >= last_month_date]

    last_week = 7
    last_week_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_week)
    data_lw = data[data[time_col] >= last_week_date]

    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']

    new_cols = []
    data_ = data[data.customer_id.isin(samples.customer_id.unique())].merge(articles, on=iid, how='left')

    for col in tqdm(cols):
        tmp = data_.groupby([uid, col])[iid].agg('count').reset_index()
        new_col = 'customer_{}_hist_cnt'.format(col)
        tmp.columns = [uid, col, new_col]
        new_cols += [new_col]
        samples = samples.merge(tmp, on=[uid, col], how='left')

    # 上次购买候选物品距今时间
    tmp = data.groupby([uid, iid])['t_dat'].agg('max').reset_index()
    tmp['purchase_corr_article_max_time'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days
    new_cols += ['purchase_corr_article_max_time']
    samples = samples.merge(tmp[[uid, iid, 'purchase_corr_article_max_time']],
                            on=[uid, iid], how='left')

    # 过去购买过该物品次数统计
    tmp = data.groupby([uid, iid])['t_dat'].agg('count').reset_index()
    tmp.columns = [uid, iid, 'purchase_corr_article_cnt']
    new_cols += ['purchase_corr_article_cnt']
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    cols = ['count']

    # 过去三天购买过的物品次数统计
    last_3days = 3  # 30
    last_3days_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_3days)
    tmp = data_lw[data_lw.t_dat >= last_3days_date].groupby([uid, iid])['price'].agg(
        cols).reset_index()
    new_col = ['customer_article_last_3days_{}'.format(col) for col in cols]
    tmp.columns = [uid, iid] + new_col
    new_cols += new_col
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    # 过去两周购买过的物品次数统计
    last_2weeks = 14
    last_2weeks_date = datetime.datetime.strptime(date_ths, '%Y-%m-%d') - datetime.timedelta(days=last_2weeks)
    tmp = data_lm[data_lm.t_dat >= last_2weeks_date].groupby([uid, iid])['price'].agg(
        cols).reset_index()
    new_col = ['customer_article_last_2weeks_{}'.format(col) for col in cols]
    tmp.columns = [uid, iid] + new_col
    new_cols += new_col
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    # 过去一个月购买过的物品次数统计
    cols = ['count']
    tmp = data_lm.groupby([uid, iid])['price'].agg(cols).reset_index()
    new_col = ['customer_article_last_month_{}'.format(col) for col in cols]
    tmp.columns = [uid, iid] + new_col
    new_cols += new_col
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    # 过去一周购买过的物品次数统计
    tmp = data_lw.groupby([uid, iid])['price'].agg(cols).reset_index()
    new_col = ['customer_article_last_week_{}'.format(col) for col in cols]
    tmp.columns = [uid, iid] + new_col
    new_cols += new_col
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    # 过去一天购买过的物品次数统计
    tmp = data_lw[data_lw.t_dat == data_lw.t_dat.max()].groupby([uid, iid])['price'].agg(
        cols).reset_index()
    new_col = ['customer_article_last_day_{}'.format(col) for col in cols]
    tmp.columns = [uid, iid] + new_col
    new_cols += new_col
    samples = samples.merge(tmp, on=[uid, iid], how='left')

    # 历史最近一次点击距今时间
    tmp = data_.groupby(uid)['t_dat'].agg('max').reset_index()
    tmp['latest_purchase_time_sub'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days
    samples = samples.merge(tmp[[uid, 'latest_purchase_time_sub']], on=uid, how='left')
    new_cols.append('latest_purchase_time_sub')

    del data_, tmp

    return samples


# def customer_feature_engineer(samples, data, uid, iid, time_col):
#     df = data[data[uid].isin(samples[uid].unique())]
#     tmp = df.groupby(uid).agg({iid: ['count', 'nunique']}).reset_index()
#     tmp.columns = [uid, 'n_purchase', 'n_purchase_nunique']
#
#     return samples

def customer_feature_engineer(samples, data, uid, iid, time_col):

    df = data[data[uid].isin(samples[uid].unique())]

    group = df.groupby('customer_id')

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

    del tmp
    gc.collect()

    return samples

# def article_feature_engineer(samples, data, uid, iid, time_col):
#     tmp = data.groupby(iid)[uid].agg('count').reset_index()
#     tmp.columns = [iid, 'purchased_cnt_global_hist']
#     samples = samples.merge(tmp, on=iid, how='left')
#
#     # 时间特征
#     cols = ['mean', 'min', 'max', 'std']
#     time_max = data[time_col].max()
#     data['pop_factor'] = (time_max - data[time_col]).dt.days
#     tmp = data.groupby(iid)['pop_factor'].agg(cols).reset_index()
#     tmp.columns = [iid] + ['article_time_{}'.format(col) for col in cols]
#     samples = samples.merge(tmp, on=iid, how='left')
#
#     del data['pop_factor'], tmp
#
#     return samples

def article_feature_engineer(samples, data, uid, iid, time_col):

    time_max = data[time_col].max()

    tmp = data.groupby('article_id')['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 'purchased_cnt_global_hist']
    samples = samples.merge(tmp, on='article_id', how='left')


    agg_cols = ['min', 'max', 'mean', 'std']
    tmp = data.groupby('article_id')['price'].agg(agg_cols).reset_index()
    tmp.columns = ['article_id'] + ['article_price_{}'.format(col) for col in agg_cols]
    samples = samples.merge(tmp, on='article_id', how='left')

    # article 时间特征
    cols = ['mean', 'sum', 'min', 'max', 'std']
    data['pop_factor'] = (time_max - data['t_dat']).dt.days
    tmp = data.groupby(['article_id'])['pop_factor'].agg(cols).reset_index()
    tmp.columns = ['article_id'] + ['article_time_{}'.format(col) for col in cols]
    samples = samples.merge(tmp, on='article_id', how='left')

    del data['pop_factor'], tmp

    gc.collect()
    return samples


def feature_engineer(samples, data, date,
                     customers, articles,
                     uid, iid, time_col, last_days=7, dtype='train'):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date).split(' ')[0]
        data_hist = data[data.t_dat <= begin_date]

        print('customer feature engineer')
        samples = customer_feature_engineer(samples, data_hist, uid, iid, time_col)

        print('article feature engineer')
        samples = article_feature_engineer(samples, data_hist, uid, iid, time_col)

        samples = samples.merge(customers, on=uid, how='left')
        samples = samples.merge(articles, on=iid, how='left')

        print('interact feature engineer')
        samples = interact_feature_engineer(samples, data_hist, customers, articles, uid, iid, time_col)
    elif dtype == 'test':

        print('customer feature engineer')
        samples = customer_feature_engineer(samples, data, uid, iid, time_col)

        print('article feature engineer')
        samples = article_feature_engineer(samples, data, uid, iid, time_col)

        samples = samples.merge(customers, on=uid, how='left')
        samples = samples.merge(articles, on=iid, how='left')

        print('interact feature engineer')
        samples = interact_feature_engineer(samples, data, customers, articles, uid, iid, time_col)

    return samples

