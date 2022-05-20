import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')


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

def get_sim_item(df, user_col, item_col, use_iif=False, time_max=None):
    from datetime import datetime
    # df.sor
    # df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    # user_time_ = df.groupby(user_col)['t_dat'].agg(list).reset_index()  # 引入时间因素
    # user_time_dict = dict(zip(user_time_[user_col], user_time_['t_dat']))

    df['date'] = (time_max - df['t_dat']).dt.days
    user_time_ = df.groupby(user_col)['date'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['date']))

    del df['date']
    gc.collect()

    user_price = df.groupby(user_col)['price'].agg(list).reset_index()
    user_price_dict = dict(zip(user_price[user_col], user_price['price']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数

    for user, items in tqdm(user_item_dict.items()):

        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            # t1 = datetime.strptime(user_time_dict[user][loc1], '%Y-%m-%d')
            for loc2, relate_item in enumerate(items):

                # if item == relate_item:
                #     continue

                # t1 = user_time_dict[user][loc1] # 点击时间提取
                # t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    t1 = user_time_dict[user][loc1]
                    t2 = user_time_dict[user][loc2]
                    if loc1 - loc2 > 0:
                        # sim_item[item][relate_item] += 1 * 0.7 * (0.8**(loc1-loc2-1)) * (1 - (t1 - t2) * 10000) / math.log(1 + len(items)) # 逆向
                        # sim_item[item][relate_item] += 0.7 * (0.8 ** (loc1 - loc2)) * (0.8 ** (t1 - t2).days) / math.log(1 + len(items))  # 逆向
                        sim_item[item][relate_item] += 0.7 * (0.8 ** (t1 - t2).days) / math.log(1 + len(items))
                    else:
                        # sim_item[item][relate_item] += 1 * 1.0 * (0.8**(loc2-loc1-1)) * (1 - (t2 - t1) * 10000) / math.log(1 + len(items)) # 正向
                        # sim_item[item][relate_item] += 1.0 * (0.8 ** (loc2 - loc1)) * (0.8 ** (t2 - t1).days) / math.log(1 + len(items))  # 正向
                        sim_item[item][relate_item] += 1.0 * (0.8 ** (t2 - t1).days) / math.log(1 + len(items))
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    # for i, related_items in tqdm(sim_item.items()):
    #     for j, cij in related_items.items():
    #         sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.5)  # 0.5
            # sim_item_corr[i][j] = cij / (item_cnt[i] ** 0.5 * item_cnt[j] ** 0.2)
            # sim_item_corr[i][j] = cij / (item_cnt[i] + 10) * (item_cnt[j] + 10)
    return sim_item_corr, user_item_dict, user_time_dict, user_price_dict


def ItemCF_Recommend(sim_item, user_item_dict, user_time_dict, user_price_dict, user_id, top_k, item_num, time_max,
                        rt_dict=False):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_times = user_time_dict[user_id]
    interacted_prices = user_price_dict[user_id]
    # time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')
    for loc, i in enumerate(interacted_items):
        if i in sim_item:
            time = interacted_times[loc] # datetime.datetime.strptime(interacted_times[loc], '%Y-%m-%d')
            price = interacted_prices[loc]
            items = sorted(sim_item[i].items(), reverse=True)[0:top_k]
            for j, wij in items:
                # if j not in interacted_items:
                rank.setdefault(j, 0)

                # rank[j] += wij * (0.8 ** ((time_max - time).days / 7)) * price

                # rank[j] += wij # * (0.8 ** ((time_max - time).days)) # * price

                rank[j] += wij * 0.8 ** time * price

    if rt_dict:
        return rank

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

def get_itemcf_recall(data, target_df, df, time_max, topk=200, rec_num=100, use_iif = False):

    import datetime

    time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')

    sim_item_corr, user_item_dict, user_time_dict, user_price_dict = get_sim_item(df, 'customer_id', 'article_id', use_iif=use_iif, time_max=time_max)

    samples = []

    target_df = target_df[target_df.customer_id.isin(data.customer_id.unique())]

    for cust, hist_arts, dates in tqdm(data[['customer_id', 'article_id', 't_dat']].values):

        rec = ItemCF_Recommend(sim_item_corr, user_item_dict, user_time_dict, user_price_dict, cust, topk, rec_num, time_max,)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'itemcf_score'])

    print(samples.shape)

    target_df['label'] = 1

    samples = samples.merge(target_df[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id'], how='left')

    samples['label'] = samples['label'].fillna(0)

    print('ItemCF recall: ', samples.shape)

    print(samples.label.mean())

    return samples

import datetime

save_path = 'itemcf_recall/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

def generate_recall_samples(data, date, last_days=7, topk=1000, recall_num=100, dtype='train', use_iif=False,
                            sim_last_days=14):

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

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    tmp = data_hist.groupby('customer_id')['t_dat'].agg('max').reset_index()

    target = target.merge(tmp, on='customer_id', how='left')

    print(last_month_date)

    target = target[target.t_dat >= last_month_date]

    del target['t_dat']

    gc.collect()

    # ItemCF进行召回
    data_hist_ = data_hist[data_hist.customer_id.isin(target.customer_id.unique())]

    df_hist = data_hist_.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data_hist_.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    sim_last_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=sim_last_days)

    samples = get_itemcf_recall(df_hist, target_df,
                                data_hist[data_hist.t_dat >= sim_last_date], begin_date, topk=topk,
                                rec_num=recall_num, use_iif=use_iif
                                )

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    print(samples.label.mean())

    reduce_mem(samples, cols=['itemcf_score', 'label'])

    samples.to_csv(save_path + '{}_cg1.csv'.format(dtype), index=False)

    print(samples.head())


def generate_recall_samples_test(data, date, cg, last_days=7, topk=1000, recall_num=100, use_iif=False, sim_last_days=14):

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    print(last_month_date)

    sim_last_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=sim_last_days)

    time_max = datetime.datetime.strptime(date, '%Y-%m-%d')

    sim_item_corr, user_item_dict, user_time_dict, user_price_dict = get_sim_item(data[data.t_dat >= sim_last_date], 'customer_id', 'article_id',
                                                                                  use_iif=use_iif, time_max=time_max)


    data_ = data[data.customer_id.isin(cg)]

    df_hist = data_.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data_.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    samples = []

    for cust, hist_arts, dates in tqdm(df_hist[['customer_id', 'article_id', 't_dat']].values):

        # rec = ItemCF_Recommend(sim_item_corr, hist_arts, dates, cust, topk, recall_num, time_max)

        rec = ItemCF_Recommend(sim_item_corr, user_item_dict, user_time_dict, user_price_dict, cust, topk, recall_num,
                               time_max, )

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'itemcf_score'])

    print(samples.shape)

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    # print(samples.label.mean())

    reduce_mem(samples, cols=['itemcf_score'])

    samples.to_csv(save_path + 'test_cg1.csv', index=False)

    print(samples.head())

sim_last_days = 30

recall_num, topk = 100, 2000

use_iif = True

print('train')
generate_recall_samples(transactions_train, '2020-09-15', last_days=7, topk=topk, recall_num=recall_num, dtype='train', use_iif=use_iif, sim_last_days=sim_last_days)

print('valid')
generate_recall_samples(transactions_train, '2020-09-22', last_days=7, topk=topk, recall_num=recall_num, dtype='valid', use_iif=use_iif, sim_last_days=sim_last_days)


def get_customer_group(users, data, date):
    import datetime

    last_days = 30

    begin_day = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    # begin_day = str(begin_day).split(' ')[0]

    df = data.groupby('customer_id')['t_dat'].agg('max').reset_index()

    users = users.merge(df, on='customer_id', how='left')

    cnt1, cnt2, cnt3 = 0, 0, 0

    cg1, cg2, cg3 = [], [], []

    for cust_id, max_date in tqdm(users.values):

        if max_date is np.nan:

            cnt3 += 1

            cg3.append(cust_id)

        elif max_date >= begin_day:

            cnt1 += 1

            cg1.append(cust_id)

        else:

            cnt2 += 1

            cg2.append(cust_id)

    print(cnt1, cnt2, cnt3)

    del users, df

    gc.collect()

    return cg1, cg2, cg3

samp_sub = pd.read_csv('sample_submission.csv')

cg1, cg2, cg3 = get_customer_group(samp_sub[['customer_id']], transactions_train, '2020-09-22')

print('test')
generate_recall_samples_test(transactions_train, '2020-09-22', cg1, last_days=7, topk=topk, recall_num=recall_num, use_iif=use_iif, sim_last_days=sim_last_days)