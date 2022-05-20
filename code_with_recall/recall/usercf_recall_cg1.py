import datetime
import gc
import math
import os
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')


transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

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

def get_user_sim(df, user_col, item_col, item_iif=False, only_sim_dict=False, item_num=2000):
    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    # time_user_ = df.groupby(item_col)['t_dat'].agg(list).reset_index()  # 引入时间因素
    # time_user_dict = dict(zip(time_user_[item_col], time_user_['t_dat']))

    sim_user = {}
    user_cnt = defaultdict(int)
    for item, users in tqdm(item_user_dict.items()):
        for loc1, u in enumerate(users):
            user_cnt[u] += 1
            sim_user.setdefault(u, {})
            for loc2, relate_user in enumerate(users):
                # if u == relate_user:
                #     continue
                # t1 = time_user_dict[item][loc1]  # 点击时间提取
                # t2 = time_user_dict[item][loc2]
                sim_user[u].setdefault(relate_user, 0)

                sim_user[u][relate_user] += 1 / math.log1p(len(users))

    # sim_user_corr = sim_user.copy()
    # for u, relate_user in tqdm(sim_user.items()):
    #     for v, cij in relate_user.items():
    #         sim_user[u][v] = cij / ((user_cnt[u] ** 0.4) * (user_cnt[v] ** 0.6))

    if only_sim_dict:
        return sim_user
    return sim_user, user_item_dict

def Usercf_Recommend(sim_user_corr, user_item_dict, user_id, top_k, item_num):
    rank = {}
    related_user = list(sim_user_corr[user_id].keys())[0:top_k]
    for u in related_user:
        for i in user_item_dict[u]:
            # if i not in user_item_dict[user_id]:
            rank.setdefault(i, 0)
            rank[i] += sim_user_corr[user_id][u]

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

def get_usercf_recall(data, target_df, df, time_max, topk=200, rec_num=100, use_iif = False):



    time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d')

    sim_user, user_item_dict = get_user_sim(df, 'customer_id', 'article_id', item_iif=False, only_sim_dict=False)

    samples = []

    target_df = target_df[target_df.customer_id.isin(data.customer_id.unique())]

    for cust, hist_arts, dates in tqdm(data[['customer_id', 'article_id', 't_dat']].values):

        rec = Usercf_Recommend(sim_user, user_item_dict, cust, topk, rec_num)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'usercf_score'])

    print(samples.shape)

    target_df['label'] = 1

    samples = samples.merge(target_df[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id'], how='left')

    samples['label'] = samples['label'].fillna(0)

    print('UserCF recall: ', samples.shape)

    print(samples.label.mean())

    return samples



save_path = 'usercf_recall/'

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

    # UserCF进行召回
    data_hist_ = data_hist[data_hist.customer_id.isin(target.customer_id.unique())]

    df_hist = data_hist_.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data_hist_.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    sim_last_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=sim_last_days)

    samples = get_usercf_recall(df_hist, target_df,
                                data_hist[data_hist.t_dat >= sim_last_date], begin_date, topk=topk,
                                rec_num=recall_num, use_iif=use_iif
                                )

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    print(samples.label.mean())

    reduce_mem(samples, cols=['usercf_score', 'label'])

    samples.to_csv(save_path + '{}_cg1.csv'.format(dtype), index=False)

    print(samples.head())


def generate_recall_samples_test(data, date, cg, last_days=7, topk=1000, recall_num=100, use_iif=False, sim_last_days=14):

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    print(last_month_date)

    sim_last_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=sim_last_days)

    time_max = datetime.datetime.strptime(date, '%Y-%m-%d')

    sim_user, user_item_dict = get_user_sim(data[data.t_dat >= sim_last_date], 'customer_id', 'article_id', item_iif=False, only_sim_dict=False)

    data_ = data[data.customer_id.isin(cg)]

    df_hist = data_.groupby('customer_id')['article_id'].agg(list).reset_index()

    tmp = data_.groupby('customer_id')['t_dat'].agg(list).reset_index()
    df_hist = df_hist.merge(tmp, on='customer_id', how='left')

    samples = []

    for cust, hist_arts, dates in tqdm(df_hist[['customer_id', 'article_id', 't_dat']].values):

        rec = Usercf_Recommend(sim_user, user_item_dict, cust, topk, recall_num)

        for k, v in rec:

            samples.append([cust, k, v])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'usercf_score'])

    print(samples.shape)

    print('customer number: ', samples.customer_id.nunique())
    print(samples.shape)
    # print(samples.label.mean())

    reduce_mem(samples, cols=['usercf_score'])

    samples.to_csv(save_path + 'test_cg1.csv', index=False)

    print(samples.head())


sim_last_days = 30

recall_num, topk = 100, 1000

use_iif = True

print('train')
generate_recall_samples(transactions_train, '2020-09-15', last_days=7, topk=topk, recall_num=recall_num, dtype='train', use_iif=use_iif, sim_last_days=sim_last_days)

print('valid')
generate_recall_samples(transactions_train, '2020-09-22', last_days=7, topk=topk, recall_num=recall_num, dtype='valid', use_iif=use_iif, sim_last_days=sim_last_days)


def get_customer_group(users, data, date):

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
