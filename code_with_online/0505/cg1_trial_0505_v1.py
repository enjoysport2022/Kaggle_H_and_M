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

from gensim.models import Word2Vec

from time import time

import datetime


def train_model(data, size=10, save_path='w2v_model/', iter=5, window=20):
    """训练模型"""
    print('Begin training w2v model')
    begin_time = time()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = Word2Vec(data, vector_size=size, window=window, min_count=0, workers=20,
                     seed=1997, epochs=iter, sg=1, hs=1, compute_loss=True,
                     # min_alpha=0.005
                     )
    print(model.get_latest_training_loss())

    # model.save(save_path + 'all_click.model')
    model.save(save_path + 'w2v.model')

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', round(run_time, 2))  # 该循环程序运行时间： 1.4201874732
    return model


def get_w2v_model(df_, date, last_days=30, size=10, iter=5, save_path='w2v_model/', window=20, new_vocab=False,
                  day_split=False):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    df = df_[(df_.t_dat <= date) & (df_.t_dat >= begin_date)]

    # df['article_id'] = df['article_id'].astype(str)

    user_item = df.groupby('customer_id')['article_id'].agg(list).reset_index()
    model = train_model(user_item['article_id'].values, size=size, iter=iter, save_path=save_path, window=window)

    # item_emb, emb_cols = train_model(user_item['item_id'].values, size=10)

    return model

def generate_w2v_feats(last_days = 180, size=5):

    w2v_model = get_w2v_model(transactions_train, '2020-09-15', size=size, last_days=last_days, )
    article_w2v_df = pd.DataFrame()
    article_w2v_df['article_id'] = w2v_model.wv.index_to_key
    w2v_vectors = pd.DataFrame(w2v_model.wv.vectors,
                               columns=['article_w2v_dim{}'.format(i) for i in range(w2v_model.wv.vector_size)])
    article_w2v_df = pd.concat([article_w2v_df, w2v_vectors], axis=1)

    return article_w2v_df

def process(customers, articles):
    cols = ['club_member_status', 'fashion_news_frequency', 'postal_code']

    for col in tqdm(cols):
        dic = dict(zip(customers[col].unique(), range(customers[col].nunique())))

        customers[col] = customers[col].map(dic)

    cols = [col for col in articles.columns if articles[col].dtype is articles['article_id'].dtype][1:-1]

    for col in tqdm(cols):
        dic = dict(zip(articles[col].unique(), range(articles[col].nunique())))

        articles[col] = articles[col].map(dic)

    articles.drop(columns='detail_desc', inplace=True)

    return customers, articles


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

    cols = ['mean', 'sum', 'min', 'max', 'std']

    data['pop_factor'] = (time_max - data['t_dat']).dt.days

    # print(data.pop_factor.describe())

    tmp = data.groupby(['article_id'])['pop_factor'].agg(cols).reset_index()

    tmp.columns = ['article_id'] + ['article_time_{}'.format(col) for col in cols]

    samples = samples.merge(tmp, on='article_id', how='left')

    del data['pop_factor'], tmp

    gc.collect()
    '''
    tmp = data.groupby('article_id')['sales_channel_id'].agg('mean').reset_index()
    tmp.columns = ['article_id', 'article_sales_channel_mean']
    samples = samples.merge(tmp, on='article_id', how='left')
    del tmp
    gc.collect()
    '''
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

    # for f in w2v_feats:
    #     df[f] = df[f].fillna(0)

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
    tmp = group.agg({
        'sales_channel_id': 'mean',
    }).rename(columns={
        'sales_channel_id': 'customer_sales_channel_mean'
    }).reset_index()
    samples = samples.merge(tmp, on='customer_id', how='left')
    '''
    del tmp

    gc.collect()

    return samples

def interact_feature_engineer(samples, data, date_ths, data_lm, data_lw):
    # cols = [f for f in articles.columns if f not in ['article_id']]
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']

    new_cols = []

    data_ = data[data.customer_id.isin(samples.customer_id.unique())].merge(articles, on='article_id', how='left')

    for col in tqdm(cols):
        tmp = data_.groupby(['customer_id', col])['article_id'].agg('count').reset_index()

        # tmp = data_.groupby(['customer_id', col])['t_dat'].agg(['count', 'max']).reset_index()

        new_col = 'customer_{}_hist_cnt'.format(col)

        tmp.columns = ['customer_id', col, new_col]

        # tmp['purchase_corr_{}_max_time'.format(col)] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['purchase_corr_{}_max_time'.format(col)]).dt.days

        new_cols += [new_col]

        samples = samples.merge(tmp, on=['customer_id', col], how='left')

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


    # 过去一个月购买过的物品次数统计

    cols = ['count']

    tmp = data_lm.groupby(['customer_id', 'article_id'])['price'].agg(cols).reset_index()

    print('data_hist check\n', tmp)

    new_col = ['customer_article_last_month_{}'.format(col) for col in cols]

    tmp.columns = ['customer_id', 'article_id'] + new_col

    new_cols += new_col

    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')

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
    '''
    cols = ['max', 'min', 'mean', 'median', 'std']

    data_['time_sub'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - data_['t_dat']).dt.days

    tmp = data_.groupby('customer_id')['time_sub'].agg(cols).reset_index()

    tmp.columns = ['customer_id'] + ['customer_hist_purchase_time_interval_{}'.format(col) for col in cols]

    samples = samples.merge(tmp, on='customer_id', how='left')

    new_cols += ['customer_hist_purchase_time_interval_{}'.format(col) for col in cols]
    '''
    del data_, tmp

    gc.collect()
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

    del data_lw_, tmp

    gc.collect()
    '''
    samples = reduce_mem(samples, new_cols)

    return samples

def gen_detail_content_recall(sim_dict, target_df, data_hist, topn=20, topk=100, prefix='detail'):
    def REC(sim_dict, hists, topn=20, topk=100):
        rank = {}
        for art in hists:
            if art not in sim_dict:
                continue
            cnt = 0
            for sart, v in sim_dict[art].items():
                # rank[sart] = rank.get(sart, 0) + v
                rank[sart] = max(rank.get(sart, 0), v)
                cnt += 1
                if cnt > topn:
                    break
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:topk]

    # sim_dict = np.load('sim_dict/art_detail_sim_dict.npy', allow_pickle=True).item()

    df = target_df.copy()
    tmp = data_hist.groupby('customer_id')['article_id'].agg(list).reset_index()
    df = df.merge(tmp, on='customer_id', how='left')

    samples = []
    for cust, label, hists in tqdm(df.values):
        if hists is np.nan:
            continue
        rec = REC(sim_dict, hists, topn, topk)
        # rec = [k for k, v in rec]
        for k, v in rec:
            if k in label:
                samples.append([cust, k, v, 1])
            else:
                samples.append([cust, k, v, 0])
    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', '{}_content_sim_score'.format(prefix), 'label'])
    print('{} content recall: '.format(prefix), samples.shape, samples.label.mean())

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

    print('待预测周正样本对数量:', len(target_df[target_df.customer_id.isin(target.customer_id.unique())].groupby(['customer_id', 'article_id'])['t_dat'].agg('count')))

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
    label_total = samples.label.sum()
    print('正样本数量: {}'.format(label_total))
    print('---')

    # 召回过去一个月购买过的项目
    # samples_pd = data_lm[data_lm.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]
    data_hist_ = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())]
    purchase_df = data_hist_.groupby('customer_id').tail(100).reset_index(drop=True)
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

    del data_lm

    gc.collect()


    samples = pd.concat([samples, samples_pd[list(samples.columns)]])
    print(samples.shape)
    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
    print(samples.shape)
    print(samples.label.mean())

    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # BinaryNet进行召回
    binary_samples = pd.read_csv('binary_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    # if dtype == 'valid':
    binary_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(binary_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('BinaryNet召回后: ', samples.label.mean())
    del binary_samples
    gc.collect()
    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # ItemCF进行召回
    itemcf_samples = pd.read_csv('itemcf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    # if dtype == 'valid':
    itemcf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(itemcf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('ItemCF召回后: ', samples.label.mean())
    del itemcf_samples
    gc.collect()
    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # UserCF进行召回
    usercf_samples = pd.read_csv('usercf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    usercf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(usercf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('UserCF召回后: ', samples.label.mean())
    del usercf_samples
    gc.collect()
    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # w2v content recall
    sim_dict = np.load('sim_dict/w2v/sim_dict_{}.npy'.format(dtype), allow_pickle=True).item()
    dc_samples = gen_detail_content_recall(sim_dict, target, data_hist_, topn=20, topk=100, prefix='w2v')
    samples = samples.merge(dc_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('w2v content召回后: ', samples.label.mean())
    del dc_samples
    gc.collect()
    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # detail content recall
    # sim_dict = np.load('sim_dict/art_detail_sim_dict_v1.npy', allow_pickle=True).item()
    sim_dict = np.load('sim_dict/art_detail_sim_dict_{}.npy'.format(dtype), allow_pickle=True).item()
    dc_samples = gen_detail_content_recall(sim_dict, target, data_hist_, topn=20, topk=100, prefix='detail')
    # samples = samples.merge(dc_samples[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id', 'label'], how='outer')
    samples = samples.merge(dc_samples,
                            on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('detail content召回后: ', samples.label.mean())
    del dc_samples
    gc.collect()
    prev_lt = label_total
    label_total = samples.label.sum()
    print('正样本数量: {}, 增加正样本数量: {}'.format(label_total, label_total - prev_lt))
    print('---')

    # samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))

    tmp = samples.groupby('customer_id')['label'].agg('sum').reset_index()
    print('召回后具有正样本的用户占比: ', len(tmp[tmp.label > 0]) / len(tmp))

    # 特征工程

    samples = customer_feature_engineer(samples, data_hist, data_lw)

    samples = article_feature_engineer(samples, data_hist, data_lw, datetime.datetime.strptime(begin_date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    samples = samples.merge(articles, on='article_id', how='left')
    # cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
    #         'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
    #         'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
    # samples = samples.merge(articles[['article_id'] + cols], on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    # tfidf_df = pd.read_csv('tfidf_feats/offline_{}.csv'.format(dtype)).drop(columns=['type'])
    # tfidf_df = pd.read_csv('tfidf_feats/v1/offline_{}.csv'.format(dtype))
    # samples = samples.merge(tfidf_df, on='customer_id', how='left')

    '''
    # w2v sub feats
    w2v_feats = [c for c in article_w2v_df.columns if c not in ['article_id']]
    cust_feats = ['customer_{}_mean'.format(c) for c in w2v_feats]
    for i, (cf, af) in enumerate(zip(cust_feats, w2v_feats)):
        samples['customer_w2v_emb_mean_sub_article_w2v_emb_dim{}'.format(i)] = samples[cf] - samples[af]
    '''

    # 过去三天popularity
    last_days = [1, 3, 7, 14, 21, 28]
    for last_day in tqdm(last_days):
        last_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_day)
        tmp = data_hist[data_hist.t_dat > last_date].groupby('article_id')['customer_id'].agg('count').reset_index()
        tmp.columns = ['article_id', 'popularity_last_{}days'.format(last_day)]
        samples = samples.merge(tmp, on='article_id', how='left')

    del tmp
    gc.collect()

    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    samples = interact_feature_engineer(samples, data_hist, begin_date, data_hist[data_hist.t_dat >= last_month_date],
                                        data_lw)

    del data_hist, data_lw, target

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

    print(importance_df.sort_values('importance', ascending=False).head(30))

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


articles = pd.read_csv('articles.csv', dtype={'article_id': str})
customers = pd.read_csv('customers.csv')
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
samp_sub = pd.read_csv('sample_submission.csv')
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

customers, articles = process(customers, articles)

last_days, size = 90, 20 # 90, 20

# article_w2v_df = generate_w2v_feats(last_days=last_days, size=size)
article_w2v_df = pd.read_csv('w2v_feats/offline_w2v_feats.csv', dtype={'article_id': str})

recall_num = 100 # 150  # 30 # 100

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
# 0505v1