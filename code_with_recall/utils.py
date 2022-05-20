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




def process(customers, articles):
    cols = ['club_member_status', 'fashion_news_frequency', 'postal_code']
    for col in tqdm(cols):
        dic = dict(zip(customers[col].unique(), range(customers[col].nunique())))
        customers[col] = customers[col].map(dic)

    customers['age_group'] = 0
    customers.loc[customers.age < 20, 'age_group'] = 1
    customers.loc[(customers.age >= 20) & (customers.age < 30), 'age_group'] = 2
    customers.loc[(customers.age >= 30) & (customers.age < 40), 'age_group'] = 3
    customers.loc[(customers.age >= 40) & (customers.age < 50), 'age_group'] = 4
    customers.loc[(customers.age >= 50) & (customers.age < 60), 'age_group'] = 5
    customers.loc[(customers.age >= 60) & (customers.age < 70), 'age_group'] = 6
    customers.loc[(customers.age >= 70) & (customers.age < 80), 'age_group'] = 7
    customers.loc[(customers.age >= 80), 'age_group'] = 8


    cols = [col for col in articles.columns if articles[col].dtype is articles['article_id'].dtype][1:-1] + ['detail_desc']
    for col in tqdm(cols):
        dic = dict(zip(articles[col].unique(), range(articles[col].nunique())))

        articles[col] = articles[col].map(dic)

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