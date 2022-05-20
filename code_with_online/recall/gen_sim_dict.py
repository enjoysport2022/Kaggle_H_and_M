import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')


# 这个也是cf，不过是直接i2i
# 之前的会综合所有交互的item，然后排序
# 这个是直接每个交互item召回一定数量这样


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

def get_sim_item(df, user_col, item_col, use_iif=False):
    from datetime import datetime
    # df.sor
    # df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['t_dat'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['t_dat']))

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
    #         sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)  # 0.5
            # sim_item_corr[i][j] = cij / (item_cnt[i] ** 0.5 * item_cnt[j] ** 0.2)
            # sim_item_corr[i][j] = cij / (item_cnt[i] + 10) * (item_cnt[j] + 10)
    return sim_item_corr, user_item_dict, user_time_dict, user_price_dict

def get_sim_item_binary(df, user_col, item_col):
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    user_time_ = df.groupby(user_col)['t_dat'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['t_dat']))

    user_price = df.groupby(user_col)['price'].agg(list).reset_index()
    user_price_dict = dict(zip(user_price[user_col], user_price['price']))

    sim_item = {}

    for item, users in tqdm(item_user_dict.items()):

        sim_item.setdefault(item, {})

        for u in users:

            tmp_len = len(user_item_dict[u])

            for relate_item in user_item_dict[u]:
                sim_item[item].setdefault(relate_item, 0)

                sim_item[item][relate_item] += 1 / (math.log(len(users) + 1) * math.log(tmp_len + 1))
                # sim_item[item][relate_item] += ratio / (math.log(len(users) + 1) * math.log(tmp_len + 1))
                # sim_item[item][relate_item] += 1 / (len(users) + 1) * (tmp_len + 1)

    return sim_item, user_item_dict, user_time_dict, user_price_dict

import datetime


def generate_sim_dict(data, date, last_days=7, dtype='train', use_iif=False):

    save_path = 'sim_dict/{}/'.format(dtype)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    hist_df = data_hist[data_hist.t_dat >= last_month_date]

    sim_item_corr_cf, _, _, _ = get_sim_item(hist_df, 'customer_id', 'article_id', use_iif=use_iif)

    sim_item_corr_binary, _, _, _ = get_sim_item_binary(hist_df, 'customer_id', 'article_id')

    np.save(save_path + 'cf.npy', sim_item_corr_cf, allow_pickle=True)

    np.save(save_path + 'bn.npy', sim_item_corr_binary, allow_pickle=True)

def generate_sim_dict_test(data, date, use_iif=False):

    save_path = 'sim_dict/test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    print(last_month_date)

    hist_df = data[data.t_dat >= last_month_date]

    sim_item_corr_cf, _, _, _ = get_sim_item(hist_df, 'customer_id', 'article_id', use_iif=use_iif)

    sim_item_corr_binary, _, _, _ = get_sim_item_binary(hist_df, 'customer_id', 'article_id')

    np.save(save_path + 'cf.npy', sim_item_corr_cf, allow_pickle=True)

    np.save(save_path + 'bn.npy', sim_item_corr_binary, allow_pickle=True)

use_iif = True

print('train')
generate_sim_dict(transactions_train, '2020-09-15', last_days=7, dtype='train', use_iif=use_iif)

print('valid')
generate_sim_dict(transactions_train, '2020-09-22', last_days=7, dtype='valid', use_iif=use_iif)

print('test')
generate_sim_dict_test(transactions_train, '2020-09-22', use_iif=use_iif)