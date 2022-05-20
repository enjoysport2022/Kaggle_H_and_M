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

    cols = [col for col in articles.columns if articles[col].dtype is articles['article_id'].dtype][1:-1] + ['detail_desc']

    dic_dict = {}

    for col in tqdm(cols):
        dic = dict(zip(articles[col].unique(), range(articles[col].nunique())))

        articles[col] = articles[col].map(dic)

        dic_dict[col] = dic

    # articles.drop(columns='detail_desc', inplace=True)

    '''
    tmp = transactions_train.merge(articles[['article_id', 'index_group_no']], on='article_id', how='left').groupby(['customer_id', 'index_group_no'])['article_id'].agg('count').reset_index()
    tmp.sort_values('article_id', ascending=False, inplace=True)
    tmp = tmp.groupby('customer_id').head(1).reset_index(drop=True)
    tmp.columns = ['customer_id', 'attribute', 'cnt']

    customers = customers.merge(tmp[['customer_id', 'attribute']], on='customer_id', how='left')
    '''

    customers['age_group'] = 0
    customers.loc[customers.age < 20, 'age_group'] = 1
    customers.loc[(customers.age >= 20) & (customers.age < 30), 'age_group'] = 2
    customers.loc[(customers.age >= 30) & (customers.age < 40), 'age_group'] = 3
    customers.loc[(customers.age >= 40) & (customers.age < 50), 'age_group'] = 4
    customers.loc[(customers.age >= 50) & (customers.age < 60), 'age_group'] = 5
    customers.loc[(customers.age >= 60) & (customers.age < 70), 'age_group'] = 6
    customers.loc[(customers.age >= 70) & (customers.age < 80), 'age_group'] = 7
    customers.loc[(customers.age >= 80), 'age_group'] = 8

    return customers, articles, dic_dict


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


def get_recall_samples(data, dtype, time_max, target_df=None, t1=10, t2=100):

    sim_dict = np.load('sim_dict/{}/cf.npy'.format(dtype), allow_pickle=True).item()

    sim_items_dict = {}
    for i, dic in sim_dict.items():
        sim_items_dict[i] = sorted(dic.items(), key=lambda d: d[1], reverse=True)

    data.sort_values('t_dat', ascending=False, inplace=True)
    data['date'] = (time_max - data['t_dat']).dt.days

    df = data.groupby('customer_id')[['article_id', 'date']].agg(list).reset_index()
    df.columns = ['customer_id', 'hist_items', 'hist_dates']
    samples = []
    for cust, hist_items, hist_dates in tqdm(df[['customer_id', 'hist_items', 'hist_dates']].values):
        n = min(t1, len(hist_items))
        topn = t2 // n
        for i, (item, date) in enumerate(zip(hist_items[:t1], hist_dates[:t1])):
            for sim_item, sim_score in sim_items_dict.get(item, [])[:topn]:
                sample = [cust, sim_item, sim_score, item, date, i]
                samples.append(sample)
    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id', 'cf_sim_score', 'source_article', 'source_interval_date', 'pre_click'])

    print(samples.shape)
    print('recall avg number: ', samples.groupby('customer_id')['article_id'].agg(len).mean())

    if target_df is not None:
        target_df['label'] = 1
        samples = samples.merge(target_df[['customer_id', 'article_id', 'label']], on=['customer_id', 'article_id'],
                                how='left')
        samples['label'] = samples['label'].fillna(0)
        samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
        print(samples.shape)
        print(samples.label.mean())
    else:
        samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
        print(samples.shape)

    return samples



def article_feature_engineer(samples, data, data_last_week, time_max):
    group = data.groupby('article_id')

    tmp = group.agg({
        'customer_id': 'count',
        # 'sales_channel_id': lambda x: sum(x == 2),
    }).rename(columns={
        'customer_id': 'n_purchased',
        # 'sales_channel_id': 'n_online_articles_art_group'
    }).reset_index()

    samples = samples.merge(tmp, on='article_id', how='left')

    tmp = group.agg({
        'customer_id': 'nunique',
        # 'sales_channel_id': lambda x: sum(x == 1),
    }).rename(columns={
        'customer_id': 'n_purchased_nuniq',
        # 'sales_channel_id': 'n_store_articles_art_group'
    }).reset_index()

    samples = samples.merge(tmp, on='article_id', how='left')


    agg_cols = ['min', 'max', 'mean', 'std', 'median', 'sum']

    tmp = data.groupby('article_id')['price'].agg(agg_cols).reset_index()

    tmp.columns = ['article_id'] + ['article_price_{}'.format(col) for col in agg_cols]

    tmp['article_price_max_minus_min'] = tmp['article_price_max'] - tmp['article_price_min']

    samples = samples.merge(tmp, on='article_id', how='left')

    # samples['purchased_nuniq_div_cnt'] = samples['n_purchased_nuniq'] / samples['n_purchased']

    tmp = group['sales_channel_id'].agg('mean').reset_index()
    tmp.columns = ['article_id', 'article_sales_channel_mean']
    samples = samples.merge(tmp, on='article_id', how='left')

    del tmp

    gc.collect()


    return samples

def customer_feature_engineer(samples, data, data_lw):

    df = data[data.customer_id.isin(samples.customer_id.unique())]

    # w2v特征用户侧聚合

    df = df.merge(article_w2v_df, on='article_id', how='left')

    w2v_feats = [c for c in article_w2v_df.columns if c not in ['article_id']]

    # for f in w2v_feats:

    #     df[f] = df[f].fillna(0)

    tmp = df.groupby('customer_id')[w2v_feats].agg('mean').reset_index()

    tmp.columns = ['customer_id'] + ['customer_{}_mean'.format(c) for c in w2v_feats]

    samples = samples.merge(tmp, on='customer_id', how='left')

    # 加sum有提升
    # tmp = df.groupby('customer_id')[w2v_feats].agg('sum').reset_index()

    # tmp.columns = ['customer_id'] + ['customer_{}_sum'.format(c) for c in w2v_feats]

    # samples = samples.merge(tmp, on='customer_id', how='left')


    df.drop(columns=w2v_feats, inplace=True)

    gc.collect()




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



    # samples['customer_purchase_nuniq_div_cnt'] = samples['n_purchase_nuniq'] / samples['n_purchase']

    tmp = group.agg({
        'sales_channel_id': 'mean',
    }).rename(columns={
        'sales_channel_id': 'customer_sales_channel_mean'
    }).reset_index()
    samples = samples.merge(tmp, on='customer_id', how='left')


    del tmp

    gc.collect()

    return samples


def interact_feature_engineer(samples, data, date_ths, data_lw):
    # cols = [f for f in articles.columns if f not in ['article_id']]
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']

    ncols = ['product_code', 'product_type_no', 'product_group_name', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']

    new_cols = []

    data_ = data[data.customer_id.isin(samples.customer_id.unique())].merge(articles[['article_id'] + cols], on='article_id', how='left')
    data_ = data_.merge(customers[['customer_id', 'age_group']], on='customer_id', how='left')
    for col in tqdm(cols):

        # tmp = data_.groupby(['customer_id', col])['price'].agg(['count', 'mean']).reset_index()
        # tmp.columns = ['customer_id', col, 'customer_{}_hist_cnt'.format(col), 'customer_{}_hist_price_mean'.format(col)]
        # new_cols += ['customer_{}_hist_cnt'.format(col), 'customer_{}_hist_price_mean'.format(col)]

        tmp = data_.groupby(['customer_id', col])['price'].agg(['count']).reset_index()

        new_col = 'customer_{}_hist_cnt'.format(col)

        tmp.columns = ['customer_id', col, new_col]

        new_cols.append(new_col)

        samples = samples.merge(tmp, on=['customer_id', col], how='left')

    data_.drop(columns=cols, inplace=True)

    # 上次购买候选物品距今时间

    tmp = data_.groupby(['customer_id', 'article_id'])['t_dat'].agg('max').reset_index()

    tmp['purchase_corr_article_max_time'] = (
                    datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days

    new_cols += ['purchase_corr_article_max_time']

    samples = samples.merge(tmp[['customer_id', 'article_id', 'purchase_corr_article_max_time']],
                                on=['customer_id', 'article_id'], how='left')

    # 过去购买过该物品次数统计
    tmp = data_.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
    tmp.columns = ['customer_id', 'article_id', 'purchase_corr_article_cnt']
    new_cols += ['purchase_corr_article_cnt']
    samples = samples.merge(tmp, on=['customer_id', 'article_id'], how='left')

    # 上周同属性人群购买候选物品次数
    cols = ['age_group', 'postal_code']
    data_lw = data_lw.merge(customers[['customer_id'] + cols], on='customer_id', how='left')
    for col in tqdm(cols):
        tmp = data_lw.groupby([col, 'article_id'])['t_dat'].agg('count').reset_index()
        tmp.columns = [col, 'article_id', 'same_{}_purchase_corr_article_cnt_lw'.format(col)]
        new_cols += ['same_{}_purchase_corr_article_cnt_lw'.format(col)]
        samples = samples.merge(tmp, on=[col, 'article_id'], how='left')
    data_lw.drop(columns=cols, inplace=True)

    # 历史最近一次点击距今时间

    tmp = data_.groupby('customer_id')['t_dat'].agg('max').reset_index()

    tmp['latest_purchase_time_sub'] = (datetime.datetime.strptime(date_ths, '%Y-%m-%d') - tmp['t_dat']).dt.days

    samples = samples.merge(tmp[['customer_id', 'latest_purchase_time_sub']], on='customer_id', how='left')

    new_cols.append('latest_purchase_time_sub')

    del data_, tmp

    gc.collect()

    # 用户过去交互序列流行度统计值
    agg_cols = ['min', 'max', 'mean', 'std']

    data['popularity'] = data.groupby(['article_id', 't_dat'])['customer_id'].transform('count')

    tmp = data.groupby('customer_id')['popularity'].agg(agg_cols).reset_index()

    tmp.columns = ['customer_id'] + ['popularity_{}'.format(col) for col in agg_cols]

    new_cols += ['popularity_{}'.format(col) for col in agg_cols]

    samples = samples.merge(tmp, on='customer_id', how='left')

    del tmp

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

    target_df = data[(data.t_dat <= date) & (data.t_dat > begin_date)]

    target = target_df.groupby('customer_id')['article_id'].agg(list).reset_index()

    target.columns = ['customer_id', 'label']

    data_hist = data[data.t_dat <= begin_date]

    tmp = data_hist.groupby('customer_id')['t_dat'].agg('max').reset_index()

    target = target.merge(tmp, on='customer_id', how='left')

    last_month_days = 30

    last_month_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_month_days)

    last_month_date = str(last_month_date).split(' ')[0]

    print(last_month_date)

    target = target[target.t_dat < last_month_date]

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

    # 召回过去半年购买过的项目

    last_hy_days = 30 * 6

    last_hy_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_hy_days)

    last_hy_date = str(last_hy_date).split(' ')[0]

    data_hy = data_hist[data_hist.t_dat >= last_hy_date]

    # samples_pd = data_hy[data_hy.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]

    # samples_pd = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]
    data_hist_ = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())]
    purchase_df = data_hist_.groupby('customer_id').tail(100).reset_index(drop=True)
    samples_pd = purchase_df[['customer_id', 'article_id']]

    pd_df = samples_pd.groupby('customer_id')['article_id'].agg(list).reset_index()

    samples_pd = []

    pd_df = pd_df.merge(target, on='customer_id', how='left')

    print('last half year shape: ', pd_df.shape)

    for cust, carts, label in tqdm(pd_df.values):

        for cart in carts:

            if cart in label:

                samples_pd.append([cust, cart, 1])

            else:

                samples_pd.append([cust, cart, 0])

    samples_pd = pd.DataFrame(samples_pd, columns=['customer_id', 'article_id', 'label'])

    # samples_pd['popularity'] = samples_pd['article_id'].map(dummy_dict)

    print(samples_pd.shape)

    # print(samples_pd.popularity.value_counts())

    del data_hy

    gc.collect()

    samples = pd.concat([samples, samples_pd[list(samples.columns)]])

    # delete duplicates samples

    print(samples.shape)

    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)

    print(samples.shape)

    # ItemCF进行召回
    print('ItemCF召回')
    cf_samples = pd.read_csv('itemcf_recall/' + '{}_cg2.csv'.format(dtype), dtype={'article_id': str})
    cf_samples = cf_samples.groupby('customer_id').head(100).reset_index(drop=True)
    cf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    # print(cf_samples.shape)
    samples = samples.merge(cf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('ItemCF召回后: ', samples.label.mean())
    del cf_samples
    gc.collect()

    # BinaryNet进行召回
    binary_samples = pd.read_csv('binary_recall/' + '{}_cg2.csv'.format(dtype), dtype={'article_id': str})
    binary_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(binary_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('BinaryNet召回后: ', samples.label.mean())
    del binary_samples
    gc.collect()

    # cate pop召回
    cp_samples = pd.read_csv('cate_pop_recall/{}.csv'.format(dtype))
    cp_samples['customer_id'] = cp_samples['customer_id'].map(cust_ex_dic)
    cp_samples['article_id'] = cp_samples['article_id'].map(art_ex_dic)
    cp_samples = cp_samples[cp_samples.customer_id.isin(samples.customer_id.unique())]
    samples = samples.merge(cp_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('Cate_Pop召回后: ', samples.label.mean())
    del cp_samples
    gc.collect()

    # cf召回
    t1, t2 = 10, 200  # 20, 300 # 20, 200
    cf_samples = get_recall_samples(data_hist_, dtype, datetime.datetime.strptime(begin_date, '%Y-%m-%d'),
                                    target_df[target_df.customer_id.isin(samples.customer_id.unique())],
                                    t1=t1, t2=t2)
    cf_samples = cf_samples.drop(columns=['source_article'])
    samples = samples.merge(cf_samples,
                            on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('cf召回后: ', samples.label.mean())
    del cf_samples
    gc.collect()


    samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))


    # 特征工程
    samples = customer_feature_engineer(samples, data_hist, data_lw)

    samples = article_feature_engineer(samples, data_hist, data_lw, datetime.datetime.strptime(begin_date, '%Y-%m-%d'))

    samples = samples.merge(customers, on='customer_id', how='left')

    # samples = samples.merge(articles, on='article_id', how='left')
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']
    samples = samples.merge(articles[['article_id'] + cols], on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    # tfidf_df = pd.read_csv('tfidf_feats/online_train.csv').drop(columns=['type'])
    tfidf_df = pd.read_csv('tfidf_feats/v1/online_train.csv')
    samples = samples.merge(tfidf_df, on='customer_id', how='left')

    del tfidf_df
    gc.collect()

    # w2v sub feats
    w2v_feats = [c for c in article_w2v_df.columns if c not in ['article_id']]
    cust_feats = ['customer_{}_mean'.format(c) for c in w2v_feats]
    for i, (cf, af) in enumerate(zip(cust_feats, w2v_feats)):
        samples['customer_w2v_emb_mean_sub_article_w2v_emb_dim{}'.format(i)] = samples[cf] - samples[af]

    # 过去一天popularity
    tmp = data_lw[data_lw.t_dat == begin_date].groupby('article_id')['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 'popularity_last_day']
    samples = samples.merge(tmp, on='article_id', how='left')

    # 过去三天popularity
    last_three_days = 3
    last_three_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_three_days)
    tmp = data_lw[data_lw.t_dat >= last_three_date].groupby('article_id')['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 'popularity_last_3days']
    samples = samples.merge(tmp, on='article_id', how='left')


    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    samples = interact_feature_engineer(samples, data_hist, begin_date, data_lw)

    del data_hist, data_lw, target

    gc.collect()

    return samples


def construct_samples_test(custs, data, date, last_days=7, recall_num=100, is_train=True):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    print(begin_date, date)

    # 热门项目召回列表
    data_lw = data[(data.t_dat >= begin_date) & (data.t_dat <= date)]

    dummy_dict = data_lw['article_id'].value_counts()

    dummy_list = [(k, v) for k, v in dummy_dict.items()][:recall_num]

    samples = []

    for cust in custs:

        for cart, pv in dummy_list:
            samples.append([cust, cart])

    samples = pd.DataFrame(samples, columns=['customer_id', 'article_id'])

    # 召回过去半年购买过的项目

    last_hy_days = 30 * 6

    last_hy_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_hy_days)

    last_hy_date = str(last_hy_date).split(' ')[0]

    # samples_pd = data_hy[data_hy.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]

    # samples_pd = data[data.customer_id.isin(samples.customer_id.unique())][['customer_id', 'article_id']]
    data_ = data[data.customer_id.isin(samples.customer_id.unique())]
    purchase_df = data_.groupby('customer_id').tail(100).reset_index(drop=True)
    samples_pd = purchase_df[['customer_id', 'article_id']]

    # samples_pd['popularity'] = samples_pd['article_id'].map(dummy_dict)

    print(samples_pd.shape)

    # print(samples_pd.popularity.value_counts())

    samples = pd.concat([samples, samples_pd[list(samples.columns)]])

    print(samples.shape)

    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)

    print(samples.shape)

    # ItemCF进行召回
    print('ItemCF召回')
    cf_samples = pd.read_csv('itemcf_recall/' + 'test_cg2.csv', dtype={'article_id': str})
    cf_samples = cf_samples[cf_samples['customer_id'].isin(custs)]
    cf_samples = cf_samples.groupby('customer_id').head(100).reset_index(drop=True)
    samples = samples.merge(cf_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del cf_samples
    gc.collect()

    # BinaryNet进行召回
    print('BinaryNet召回')
    binary_samples = pd.read_csv('binary_recall/' + 'test_cg2.csv', dtype={'article_id': str})
    binary_samples = binary_samples[binary_samples['customer_id'].isin(custs)]
    samples = samples.merge(binary_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del binary_samples
    gc.collect()

    # cate pop召回
    print('CatePop召回')
    cp_samples = pd.read_csv('cate_pop_recall/test.csv')
    cp_samples['customer_id'] = cp_samples['customer_id'].map(cust_ex_dic)
    cp_samples['article_id'] = cp_samples['article_id'].map(art_ex_dic)
    cp_samples = cp_samples[cp_samples.customer_id.isin(samples.customer_id.unique())]
    samples = samples.merge(cp_samples, on=['customer_id', 'article_id'], how='outer')
    print(samples.shape)
    del cp_samples
    gc.collect()

    # cf召回
    t1, t2 = 10, 200  # 20, 300 # 20, 200
    cf_samples = get_recall_samples(data_, 'test', datetime.datetime.strptime(date, '%Y-%m-%d'),
                                    target_df=None, t1=t1, t2=t2)
    cf_samples = cf_samples.drop(columns=['source_article'])
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

    # samples = samples.merge(articles, on='article_id', how='left')
    cols = ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'] + ['detail_desc']
    samples = samples.merge(articles[['article_id'] + cols], on='article_id', how='left')


    samples = samples.merge(article_w2v_df, on='article_id', how='left')

    # tfidf_df = pd.read_csv('tfidf_feats/online_valid.csv').drop(columns=['type'])
    tfidf_df = pd.read_csv('tfidf_feats/v1/online_valid.csv')
    samples = samples.merge(tfidf_df, on='customer_id', how='left')

    del tfidf_df
    gc.collect()

    # w2v sub feats
    w2v_feats = [c for c in article_w2v_df.columns if c not in ['article_id']]
    cust_feats = ['customer_{}_mean'.format(c) for c in w2v_feats]
    for i, (cf, af) in enumerate(zip(cust_feats, w2v_feats)):
        samples['customer_w2v_emb_mean_sub_article_w2v_emb_dim{}'.format(i)] = samples[cf] - samples[af]

    # 过去一天popularity
    tmp = data_lw[data_lw.t_dat == date].groupby('article_id')['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 'popularity_last_day']
    samples = samples.merge(tmp, on='article_id', how='left')

    # 过去三天popularity
    last_three_days = 3
    last_three_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_three_days)
    tmp = data_lw[data_lw.t_dat >= last_three_date].groupby('article_id')['customer_id'].agg('count').reset_index()
    tmp.columns = ['article_id', 'popularity_last_3days']
    samples = samples.merge(tmp, on='article_id', how='left')

    samples = reduce_mem(samples, [c for c in samples.columns if c not in ['customer_id', 'article_id']])

    samples = interact_feature_engineer(samples, data, date, data_lw)

    del data_lw, samples_pd

    gc.collect()

    return samples

cust_dic = np.load('map_dict/cust.npy', allow_pickle=True).item()
cust_ex_dic = {}
for k, v in cust_dic.items():
    cust_ex_dic[v] = k
art_dic = np.load('map_dict/art.npy', allow_pickle=True).item()
art_ex_dic = {}
for k, v in art_dic.items():
    art_ex_dic[v] = k

articles = pd.read_csv('articles.csv', dtype={'article_id': str})
customers = pd.read_csv('customers.csv')
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
samp_sub = pd.read_csv('sample_submission.csv')

customers, articles, dic_dict = process(customers, articles)

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

epoch = 1525

model, feats = ranker(train, epoch, dtype='cg2')


def model_inference(model, data, feats, split_num=3):

    custs = list(data['customer_id'].unique())

    bs = len(custs) // split_num

    prob = []

    for i in tqdm(range(split_num + 1)):

        tmp_custs = custs[i * bs: (i + 1) * bs]

        if not tmp_custs:

            break

        valid = data[data.customer_id.isin(tmp_custs)]

        valid['prob'] = model.predict(valid[feats])

        prob.append(valid[['customer_id', 'article_id', 'prob']])

    prob = pd.concat(prob)

    return prob

# Inference
def Inference(model, feats, transactions_train, cg, batch_size=30000, split_num=3):
    batch_num = len(cg) // batch_size + 1

    recs = []

    for i in range(batch_num):
        print('[{}/{}]'.format(i + 1, batch_num))

        custs = cg[i * batch_size: (i + 1) * batch_size]

        test = construct_samples_test(custs, transactions_train, '2020-09-22', last_days=7, recall_num=recall_num)

        print(test.shape)

        # test['prob'] = model.predict(test[feats])

        test = model_inference(model, test, feats, split_num=split_num)

        rec = test.sort_values('prob', ascending=False).groupby('customer_id')['article_id'].agg(
            lambda x: ' '.join(list(x)[:12])).reset_index()

        rec.columns = ['customer_id', 'prediction']

        recs.append(rec)

        del rec, test

        gc.collect()

    recs = pd.concat(recs)

    return recs

bs = 80000

recs_cg2 = Inference(model, feats, transactions_train, cg2, batch_size=bs)

print(recs_cg2.shape)

print(recs_cg2.head())

save_path = 'sub/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

recs_cg2.to_csv(save_path + 'sub_cg2.csv', index=False)