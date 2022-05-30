import numpy as np
import pandas as pd
from collections import defaultdict
import math
import os
import random
from tqdm import tqdm
import gc
import datetime
import warnings
warnings.filterwarnings('ignore')

# autox_recommend, recall, usercf.

def get_user_sim(df, user_col, item_col, item_iif=False, only_sim_dict=False, item_num=2000):
    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    sim_user = {}
    user_cnt = defaultdict(int)
    for item, users in tqdm(item_user_dict.items()):
        for loc1, u in enumerate(users):
            user_cnt[u] += 1
            sim_user.setdefault(u, {})
            for loc2, relate_user in enumerate(users):
                sim_user[u].setdefault(relate_user, 0)
                sim_user[u][relate_user] += 1 / math.log1p(len(users))

    if only_sim_dict:
        return sim_user
    return sim_user, user_item_dict

def Usercf_Recommend(sim_user_corr, user_item_dict, user_id, top_k, item_num):
    rank = {}
    related_user = list(sim_user_corr[user_id].keys())[0:top_k]
    for u in related_user:
        for i in user_item_dict[u]:
            rank.setdefault(i, 0)
            rank[i] += sim_user_corr[user_id][u]

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

def get_usercf_recall(data, target_df, df, time_max, topk=200, rec_num=100, use_iif = False):

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

def usercf_recall(uids, data, date, uid, iid, time_col, last_days=7, recall_num=100, dtype='train',
                  topk=1000, use_iif=False, sim_last_days=14):

    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date).split(' ')[0]

        target_df = data[(data.t_dat <= date) & (data.t_dat > begin_date)]
        target = target_df.groupby('customer_id')['article_id'].agg(list).reset_index()
        target.columns = ['customer_id', 'label']

        data_hist = data[data.t_dat <= begin_date]

        data_hist_ = data_hist[data_hist.customer_id.isin(target.customer_id.unique())]

        df_hist = data_hist_.groupby('customer_id')['article_id'].agg(list).reset_index()

        tmp = data_hist_.groupby('customer_id')['t_dat'].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on='customer_id', how='left')

        samples = get_usercf_recall(df_hist, target_df,
                                    data_hist, begin_date, topk=topk,
                                    rec_num=recall_num, use_iif=use_iif)

        return samples

    elif dtype == 'test':

        sim_user, user_item_dict = get_user_sim(data, 'customer_id', 'article_id',
                                                item_iif=False, only_sim_dict=False)

        data_ = data[data.customer_id.isin(uids)]
        df_hist = data_.groupby('customer_id')['article_id'].agg(list).reset_index()

        tmp = data_.groupby('customer_id')['t_dat'].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on='customer_id', how='left')

        samples = []
        for cust, hist_arts, dates in tqdm(df_hist[['customer_id', 'article_id', 't_dat']].values):
            if cust not in user_item_dict:
                continue
            rec = Usercf_Recommend(sim_user, user_item_dict, cust, topk, recall_num)
            for k, v in rec:
                samples.append([cust, k, v])
        samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'usercf_score'])

        return samples