import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')

customers = pd.read_csv('customers.csv')

articles = pd.read_csv('articles.csv', dtype={'article_id': str})

save_path = 'map_dict/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
cust_dict = dict(zip(customers.customer_id.unique(), range(customers.customer_id.nunique())))
np.save(save_path + 'cust.npy', cust_dict, allow_pickle=True)
art_dict = dict(zip(articles.article_id.unique(), range(articles.article_id.nunique())))
np.save(save_path + 'art.npy', art_dict, allow_pickle=True)

transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})

transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

from collections import defaultdict

import math


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


def gen_rec(hist_arts, cate_pop_dict, art_cate_dict, rec_num=12):
    rec = {}
    # print(hist_cates, hist_cnts)
    for art in hist_arts:

        for c, pop_dict in cate_pop_dict.items():

            # w = cate_weights[c]

            ac_dic = art_cate_dict[c]

            cate = ac_dic[art]

            for sim_art, pop in pop_dict.get(cate, {}).items():

                rec[sim_art] = rec.get(sim_art, 0) + pop  # * w

    return sorted(rec.items(), key=lambda d: d[1], reverse=True)[:rec_num]


def get_recall(data, target_df, cate_pop_dict, art_cate_dict, rec_num=100):

    import datetime

    samples = []

    target_df = target_df[target_df.customer_id.isin(data.customer_id.unique())]

    # time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')

    for cust, hist_arts, dates in tqdm(data[['customer_id', 'article_id', 't_dat']].values):

        rec = gen_rec(hist_arts, cate_pop_dict, art_cate_dict, rec_num=rec_num)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'cate_pop_score'])

    print(samples.shape)

    target_df['label'] = 1

    samples = samples.merge(target_df[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id'], how='left')

    samples['label'] = samples['label'].fillna(0)

    print('cate pop recall: ', samples.shape)

    # print(samples.label.mean())

    return samples

import datetime

save_path = 'cate_pop_recall/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

def generate_recall_samples(data, date, cols, last_days=7, topk=100, recall_num=100, dtype='train'):

    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    pop_begin_date = str(pop_begin_date).split(' ')[0]

    print(pop_begin_date, begin_date, date)

    # 热门项目召回列表
    target_df = data[(data.t_dat <= date) & (data.t_dat > begin_date)]
    target = target_df.groupby('customer_id')['article_id'].agg(list).reset_index()
    target.columns = ['customer_id', 'label']

    data_hist = data[data.t_dat <= begin_date]

    # 流行度
    pop_df = data_hist[data_hist.t_dat >= pop_begin_date].groupby('article_id')['customer_id'].agg('count').reset_index()
    pop_df.columns = ['article_id', 'popularity']
    pop_df = pop_df.merge(articles[['article_id'] + cols], on='article_id', how='left')
    pop_df.sort_values('popularity', ascending=False, inplace=True)
    print(pop_df.head())
    print(pop_df.shape)
    cate_pop_dict = {}
    for col in cols:
        pop_dict = {}
        for cate in tqdm(pop_df[col].unique()):
            df = pop_df[pop_df[col] == cate]
            pop_dict[cate] = dict(zip(df['article_id'].values[:topk], df['popularity'].values[:topk]))
        cate_pop_dict[col] = pop_dict

    # 召回
    data_hist_ = data_hist[data_hist.customer_id.isin(target.customer_id.unique())]

    df_hist = data_hist_.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data_hist_.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    samples = get_recall(df_hist, target_df, cate_pop_dict,
                         art_cate_dict, rec_num=recall_num
                         )

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    print(samples.shape)
    print(samples.label.mean())

    samples['customer_id'] = samples['customer_id'].map(cust_dict)
    samples['article_id'] = samples['article_id'].map(art_dict)

    reduce_mem(samples, cols=['customer_id', 'article_id', 'cate_pop_score', 'label'])

    samples.to_csv(save_path + '{}.csv'.format(dtype), index=False)

    print(samples.head())


def generate_recall_samples_test(data, date, cols, last_days=7, topk=100, recall_num=100):

    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]


    time_max = datetime.datetime.strptime(date, '%Y-%m-%d')


    df_hist = data.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    # 流行度
    pop_df = data[data.t_dat >= begin_date].groupby('article_id')['customer_id'].agg('count').reset_index()
    pop_df.columns = ['article_id', 'popularity']
    pop_df = pop_df.merge(articles[['article_id'] + cols], on='article_id', how='left')
    pop_df.sort_values('popularity', ascending=False, inplace=True)
    print(pop_df.head())
    print(pop_df.shape)
    cate_pop_dict = {}
    for col in cols:
        pop_dict = {}
        for cate in tqdm(pop_df[col].unique()):
            df = pop_df[pop_df[col] == cate]
            pop_dict[cate] = dict(zip(df['article_id'].values[:topk], df['popularity'].values[:topk]))
        cate_pop_dict[col] = pop_dict

    samples = []

    for cust, hist_arts, dates in tqdm(df_hist[['customer_id', 'article_id', 't_dat']].values):

        rec = gen_rec(hist_arts, cate_pop_dict, art_cate_dict, rec_num=recall_num)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'cate_pop_score'])

    print(samples.shape)

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)

    samples['customer_id'] = samples['customer_id'].map(cust_dict)
    samples['article_id'] = samples['article_id'].map(art_dict)

    reduce_mem(samples, cols=['customer_id', 'article_id', 'cate_pop_score'])

    samples.to_csv(save_path + 'test.csv', index=False)

    print(samples.head())

recall_num, topk = 100, 100

use_cols = ['product_group_name', 'department_no', 'section_no']
art_cate_dict = {}
for col in use_cols:
    art_cate_dict[col] = dict(zip(articles['article_id'].values, articles[col].values))

print('train')
generate_recall_samples(transactions_train, '2020-09-15', use_cols, last_days=7, topk=topk, recall_num=recall_num, dtype='train')

print('valid')
generate_recall_samples(transactions_train, '2020-09-22', use_cols, last_days=7, topk=topk, recall_num=recall_num, dtype='valid')

samp_sub = pd.read_csv('sample_submission.csv')

print('test')
generate_recall_samples_test(transactions_train, '2020-09-22', use_cols, last_days=7, topk=topk, recall_num=recall_num)
