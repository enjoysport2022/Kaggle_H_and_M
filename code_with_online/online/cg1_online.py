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

    cols = [col for col in articles.columns if articles[col].dtype is articles['article_id'].dtype][1:-1]

    for col in tqdm(cols):
        dic = dict(zip(articles[col].unique(), range(articles[col].nunique())))

        articles[col] = articles[col].map(dic)

    articles.drop(columns='detail_desc', inplace=True)

    return customers, articles


def get_customer_group(users, data, date):
    import datetime

    last_days = 30
    begin_day = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
    begin_day = str(begin_day).split(' ')[0]

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


import datetime


def article_feature_engineer(samples, data, data_last_week, time_max):
    art_feats = data.groupby('article_id')['customer_id'].agg('count').reset_index()

    art_feats.columns = ['article_id', 'purchased_cnt_global_hist']

    agg_cols = ['min', 'max', 'mean', 'std']

    # tmp = data.groupby('article_id')['price'].agg(agg_cols).reset_index()

    # tmp.columns = ['article_id'] + ['article_price_{}'.format(col) for col in agg_cols]

    # art_feats = art_feats.merge(tmp, on='article_id', how='left')

    tmp = data.groupby('article_id')['price'].agg(agg_cols).reset_index()

    tmp.columns = ['article_id'] + ['article_price_{}'.format(col) for col in agg_cols]

    art_feats = art_feats.merge(tmp, on='article_id', how='left')

    # article 时间特征

    cols = ['mean', 'sum', 'min', 'max', 'std']

    data['pop_factor'] = (time_max - data['t_dat']).dt.days

    # print(data.pop_factor.describe())

    tmp = data.groupby(['article_id'])['pop_factor'].agg(cols).reset_index()

    tmp.columns = ['article_id'] + ['article_time_{}'.format(col) for col in cols]

    art_feats = art_feats.merge(tmp, on='article_id', how='left')

    del data['pop_factor']

    gc.collect()

    samples = samples.merge(art_feats, on='article_id', how='left')

    del art_feats, tmp

    gc.collect()

    return samples

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

    # 过去购买过该物品次数统计
    tmp = data.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
    tmp.columns = ['customer_id', 'article_id', 'purchase_corr_article_cnt']
    new_cols += ['purchase_corr_article_cnt']
    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')

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

    samples = reduce_mem(samples, new_cols)

    return samples

def construct_samples(data, date, last_days=7, recall_num=100, dtype='valid'):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    pop_begin_date = str(pop_begin_date).split(' ')[0]

    print(pop_begin_date, begin_date, date)

    # 热门项目召回列表
    data_lw = data[(data.t_dat >= pop_begin_date) & (data.t_dat <= begin_date)]

    dummy_dict = data_lw['article_id'].value_counts()

    dummy_list = [(k, v) for k, v in dummy_dict.items()][:recall_num]

    target = data[(data.t_dat <= date) & (data.t_dat > begin_date)]

    target = target.groupby('customer_id')['article_id'].agg(list).reset_index()

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
    # samples_pd = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]
    purchase_df = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())].groupby('customer_id').tail(
        100).reset_index(drop=True)
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

    # BinaryNet进行召回
    print('BinaryNet召回')
    binary_samples = pd.read_csv('binary_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    binary_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(binary_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('BinaryNet召回后: ', samples.label.mean())
    del binary_samples
    gc.collect()

    # ItemCF进行召回
    print('ItemCF召回')
    itemcf_samples = pd.read_csv('itemcf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
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


    samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))


    # 特征工程

    samples = customer_feature_engineer(samples, data_hist, data_lw)

    samples = article_feature_engineer(samples, data_hist, data_lw, datetime.datetime.strptime(begin_date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    samples = samples.merge(articles, on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    tfidf_df = pd.read_csv('tfidf_feats/online_train.csv').drop(columns=['type'])
    samples = samples.merge(tfidf_df, on='customer_id', how='left')


    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])


    samples = interact_feature_engineer(samples, data_hist, begin_date, data_hist[data.t_dat >= last_month_date],
                                        data_lw)

    del data_hist, data_lw, target

    gc.collect()

    return samples


def construct_samples_test(custs, data, date, last_days=7, recall_num=100):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    print(begin_date, date)

    # 热门项目召回列表
    data_lw = data[(data.t_dat >= begin_date) & (data.t_dat <= date)]

    dummy_dict = data_lw['article_id'].value_counts()

    dummy_list = [(k, v) for k, v in dummy_dict.items()][:recall_num]

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    data_lm = data[data.t_dat >= last_month_date]

    samples = []

    for cust in custs:

        for cart, pv in dummy_list:
            samples.append([cust, cart])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id'])

    # 召回过去所有购买过的项目
    purchase_df = data[data.customer_id.isin(samples.customer_id.unique())].groupby('customer_id').tail(
        100).reset_index(drop=True)
    samples_pd = purchase_df[['customer_id', 'article_id']]

    del purchase_df
    gc.collect()
    # samples_pd['popularity'] = samples_pd['article_id'].map(dummy_dict)

    print(samples_pd.shape)

    # print(samples_pd.popularity.value_counts())

    del data_lm

    gc.collect()

    samples = pd.concat([samples, samples_pd[list(samples.columns)]])

    del samples_pd

    gc.collect()

    print(samples.shape)

    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)

    print(samples.shape)

    # BinaryNet进行召回
    print('BinaryNet召回')
    binary_samples = pd.read_csv('binary_recall/' + 'test_cg1.csv', dtype={'article_id': str})
    binary_samples = binary_samples[binary_samples['customer_id'].isin(custs)]

    samples = samples.merge(binary_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del binary_samples
    gc.collect()

    # ItemCF进行召回
    print('ItemCF召回')
    cf_samples = pd.read_csv('itemcf_recall/' + 'test_cg1.csv', dtype={'article_id': str})
    cf_samples = cf_samples[cf_samples['customer_id'].isin(custs)]

    samples = samples.merge(cf_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del cf_samples
    gc.collect()

    # UserCF进行召回
    print('UserCF召回')
    cf_samples = pd.read_csv('usercf_recall/' + 'test_cg1.csv', dtype={'article_id': str})
    cf_samples = cf_samples[cf_samples['customer_id'].isin(custs)]

    samples = samples.merge(cf_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del cf_samples
    gc.collect()




    samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))


    # 特征工程

    samples = customer_feature_engineer(samples, data, data_lw)

    samples = article_feature_engineer(samples, data, data_lw, datetime.datetime.strptime(date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    samples = samples.merge(articles, on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    tfidf_df = pd.read_csv('tfidf_feats/online_valid.csv').drop(columns=['type'])
    samples = samples.merge(tfidf_df, on='customer_id', how='left')
    del tfidf_df
    gc.collect()

    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    samples = interact_feature_engineer(samples, data, date, data[data.t_dat >= last_month_date], data_lw)

    del data_lw

    gc.collect()

    return samples

articles = pd.read_csv('articles.csv', dtype={'article_id': str})
customers = pd.read_csv('customers.csv')
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
samp_sub = pd.read_csv('sample_submission.csv')

customers, articles = process(customers, articles)

article_w2v_df = pd.read_csv('w2v_feats/online_w2v_feats.csv', dtype={'article_id': str})

cg1, cg2, cg3 = get_customer_group(samp_sub[['customer_id']], transactions_train, '2020-09-22')


transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

recall_num = 100

train = construct_samples(transactions_train, '2020-09-22', last_days=7, recall_num=recall_num, dtype='valid')

import lightgbm as lgb

def ranker(train, epoch, dtype):
    train.sort_values(by=['customer_id'], inplace=True)

    g_train = train.groupby(['customer_id'], as_index=False).count()["label"].values

    feats = [f for f in train.columns if f not in ['customer_id', 'article_id', 'label', 'prob']]

    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=epoch, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1)

    lgb_ranker.fit(train[feats], train['label'], group=g_train,
                   eval_set=[(train[feats], train['label'])],
                   eval_group=[g_train], eval_at=[12], eval_metric=['map', ],
                   verbose=100
                   )

    print(lgb_ranker.best_score_)

    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = lgb_ranker.feature_importances_

    del train

    gc.collect()

    print(importance_df.sort_values('importance', ascending=False).head(20))

    return lgb_ranker, feats

epoch = 932

model, feats = ranker(train, epoch, dtype='cg1')


# Inference
def Inference(model, feats, transactions_train, batch_size=30000):
    batch_num = len(cg1) // batch_size + 1

    recs = []

    for i in range(batch_num):
        print('[{}/{}]'.format(i + 1, batch_num))

        custs = cg1[i * batch_size: (i + 1) * batch_size]

        test = construct_samples_test(custs, transactions_train, '2020-09-22', last_days=7, recall_num=recall_num)

        print(test.shape)

        test['prob'] = model.predict(test[feats])

        rec = test.sort_values('prob', ascending=False).groupby('customer_id')['article_id'].agg(
            lambda x: ' '.join(list(x)[:12])).reset_index()

        rec.columns = ['customer_id', 'prediction']

        recs.append(rec)

        del rec

        gc.collect()

    recs = pd.concat(recs)

    return recs

bs = 60000

recs_cg1 = Inference(model, feats, transactions_train, batch_size=bs)

print(recs_cg1.shape)

print(recs_cg1.head())

save_path = 'sub/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

recs_cg1.to_csv(save_path + 'sub_cg1.csv', index=False)
