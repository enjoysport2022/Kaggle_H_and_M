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

def get_sim_item_binary(df, user_col, item_col, time_max):
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    df['date'] = (time_max - df['t_dat']).dt.days

    # user_time_ = df.groupby(user_col)['t_dat'].agg(list).reset_index()  # 引入时间因素
    # user_time_dict = dict(zip(user_time_[user_col], user_time_['t_dat']))
    user_time_ = df.groupby(user_col)['date'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['date']))

    del df['date']

    gc.collect()

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
                # sim_item[item][relate_item] += 1
                # sim_item[item][relate_item] += ratio / (math.log(len(users) + 1) * math.log(tmp_len + 1))
                # sim_item[item][relate_item] += 1 / (len(users) + 1) * (tmp_len + 1)

    return sim_item, user_item_dict, user_time_dict, user_price_dict


def BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, user_price_dict, user_id, top_k, item_num, time_max,
                        rt_dict=False):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_times = user_time_dict[user_id]
    # interacted_prices = user_price_dict[user_id]
    # time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')
    for loc, i in enumerate(interacted_items):
        time = interacted_times[loc] # datetime.datetime.strptime(interacted_times[loc], '%Y-%m-%d')
        # price = interacted_prices[loc]
        items = sorted(sim_item[i].items(), reverse=True)[0:top_k]
        for j, wij in items:
            # if j not in interacted_items:
            rank.setdefault(j, 0)

            # rank[j] += wij * (0.8 ** ((time_max - time).days / 7)) * price

            # rank[j] += wij # * (0.8 ** ((time_max - time).days)) # * price

            rank[j] += wij * 0.8 ** time # * price

    if rt_dict:
        return rank

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

def get_binaryNet_recall(custs, target_df, df, time_max, topk=200, rec_num=100):

    import datetime

    time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')

    sim_item, user_item_dict, user_time_dict, user_price_dict = get_sim_item_binary(df, 'customer_id', 'article_id', time_max)

    samples = []

    target_df = target_df[target_df.customer_id.isin(custs)]

    for cust in tqdm(custs):

        rec = BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, user_price_dict, cust, topk, rec_num,
                                  time_max)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'binary_score'])

    print(samples.shape)

    target_df['label'] = 1

    samples = samples.merge(target_df[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id'], how='left')

    samples['label'] = samples['label'].fillna(0)

    print('BinaryNet recall: ', samples.shape)

    print(samples.label.mean())

    return samples

import datetime

save_path = 'binary_recall/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

def generate_recall_samples(data, date, last_days=7, topk=1000, recall_num=100, dtype='train'):

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

    # BinaryNet进行召回

    binary_samples = get_binaryNet_recall(target.customer_id.unique(), target_df,
                                          data_hist[data_hist.t_dat >= last_month_date], begin_date, topk=topk,
                                          rec_num=recall_num)

    print('customer number: ', binary_samples.customer_id.nunique())
    print(binary_samples.shape)
    print(binary_samples.label.mean())

    binary_samples.to_csv(save_path + '{}_cg1.csv'.format(dtype), index=False)

    print(binary_samples.head())


def generate_recall_samples_test(data, date, cg, last_days=7, topk=1000, recall_num=100):

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    print(last_month_date)

    time_max = datetime.datetime.strptime(date, '%Y-%m-%d')

    sim_item, user_item_dict, user_time_dict, user_price_dict = get_sim_item_binary(data[data.t_dat >= last_month_date], 'customer_id', 'article_id', time_max)

    samples = []

    for cust in tqdm(cg):

        if cust in user_item_dict:

            rec = BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, user_price_dict, cust, topk, recall_num,
                                      time_max)

            for k, v in rec:

                samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'binary_score'])

    print(samples.shape)

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    # print(samples.label.mean())

    samples.to_csv(save_path + 'test_cg1.csv', index=False)

    print(samples.head())

recall_num, topk = 100, 1000
# recall_num, topk = 200, 2000

print('train')
generate_recall_samples(transactions_train, '2020-09-15', last_days=7, topk=topk, recall_num=recall_num, dtype='train')

print('valid')
generate_recall_samples(transactions_train, '2020-09-22', last_days=7, topk=topk, recall_num=recall_num, dtype='valid')


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
generate_recall_samples_test(transactions_train, '2020-09-22', cg1, last_days=7, topk=topk, recall_num=recall_num)
