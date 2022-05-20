import datetime
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc

def construct_samples(data, date, customers, cust_ex_dic,  art_ex_dic, last_days=7, recall_num=100, dtype='train'):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
    pop_begin_date = str(pop_begin_date).split(' ')[0]
    print(pop_begin_date, begin_date, date)

    # 召回1: 热门项目召回列表
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



    # # 增加召回: 每个年龄最热门的100个article_id
    # tmp = data_lw[['t_dat', 'customer_id', 'article_id']].merge(customers[['customer_id', 'age']],
    #                                                               on='customer_id', how='left')
    # age_popular = {}
    # for cur_age in tqdm(tmp['age'].unique()):
    #     try:
    #         age_popular[int(cur_age)] = tmp.loc[tmp['age'] == cur_age]['article_id'].value_counts().index[:100]
    #     except:
    #         age_popular[-1] = tmp.loc[tmp['age'].isnull()]['article_id'].value_counts().index[:100]
    #
    # cust2age = customers.drop_duplicates("customer_id", keep='last')[['customer_id', 'age']].fillna(-1)
    # cust2age['age'] = cust2age['age'].astype(int)
    # cust2age = cust2age.set_index(["customer_id"])['age'].to_dict()
    #
    # samples_pd = []
    # for cust, label in target.values:
    #     age = cust2age[cust]
    #     carts = age_popular[age]
    #     for cart in carts:
    #         if cart in label:
    #             samples_pd.append([cust, cart, 1])
    #         else:
    #             samples_pd.append([cust, cart, 0])
    # samples_pd = pd.DataFrame(samples_pd, columns=['customer_id', 'article_id', 'label'])
    # print(samples_pd.shape)
    #
    # samples = pd.concat([samples, samples_pd[list(samples.columns)]])
    # print(samples.shape)
    # samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
    # print(samples.shape)
    # print(samples.label.mean())



    # 增加召回: 每个价格区间最热门的100个article_id
    def price2cat(x):
        #     [-0.0009831, 0.0101, 0.0135, 0.0169, 0.0203, 0.0254, 0.0271, 0.0339, 0.0381, 0.0508, 0.592]
        for i, up in enumerate([0.0101, 0.0135, 0.0169, 0.0203, 0.0254, 0.0271, 0.0339, 0.0381, 0.0508, 0.592]):
            if x < up:
                return i

    tmp = data_hist[['t_dat', 'customer_id', 'article_id', 'price']]
    tmp['cat'] = tmp['price'].apply(lambda x: price2cat(x))
    price_popular = {}
    for cur_cat in tqdm(tmp['cat'].unique()):
        price_popular[cur_cat] = list(tmp.loc[tmp['cat'] == cur_cat]['article_id'].value_counts().index[:100])

    cust2price = tmp.groupby('customer_id').agg({'price': ['mean']}).reset_index()
    cust2price.columns = ['customer_id', 'price']
    cust2price['cat'] = cust2price['price'].apply(lambda x: price2cat(x))
    cust2price = cust2price.set_index(["customer_id"])['cat'].to_dict()

    samples_pd = []
    for cust, label in target.values:
        price = cust2price[cust]
        carts = price_popular[price]
        for cart in carts:
            if cart in label:
                samples_pd.append([cust, cart, 1])
            else:
                samples_pd.append([cust, cart, 0])
    samples_pd = pd.DataFrame(samples_pd, columns=['customer_id', 'article_id', 'label'])
    print(samples_pd.shape)

    samples = pd.concat([samples, samples_pd[list(samples.columns)]])
    print(samples.shape)
    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
    print(samples.shape)
    print(samples.label.mean())




    # 召回2: 召回过去一个月购买过的项目
    purchase_df = data_hist[data_hist.customer_id.isin(samples.customer_id.unique())].groupby('customer_id').tail(100).reset_index(drop=True)
    samples_pd = purchase_df[['customer_id', 'article_id']]
    pd_df = samples_pd.groupby('customer_id')['article_id'].agg(list).reset_index()
    pd_df = pd_df.merge(target, on='customer_id', how='left')

    samples_pd = []
    for cust, carts, label in pd_df.values:
        for cart in carts:
            if cart in label:
                samples_pd.append([cust, cart, 1])
            else:
                samples_pd.append([cust, cart, 0])
    samples_pd = pd.DataFrame(samples_pd, columns=['customer_id', 'article_id', 'label'])
    print(samples_pd.shape)

    samples = pd.concat([samples, samples_pd[list(samples.columns)]])
    print(samples.shape)
    samples.drop_duplicates(subset=['customer_id', 'article_id'], keep='first', inplace=True)
    print(samples.shape)
    print(samples.label.mean())


    # 召回3: BinaryNet进行召回
    binary_samples = pd.read_csv('binary_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    binary_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)

    samples = samples.merge(binary_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('BinaryNet召回后: ', samples.label.mean())
    del binary_samples
    gc.collect()

    # 召回4: ItemCF进行召回
    itemcf_samples = pd.read_csv('itemcf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    itemcf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)

    samples = samples.merge(itemcf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('ItemCF召回后: ', samples.label.mean())
    del itemcf_samples
    gc.collect()

    # 召回5: UserCF进行召回
    usercf_samples = pd.read_csv('usercf_recall/' + '{}_cg1.csv'.format(dtype), dtype={'article_id': str})
    usercf_samples.drop_duplicates(subset=['customer_id', 'article_id', 'label'], keep='first', inplace=True)
    samples = samples.merge(usercf_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('UserCF召回后: ', samples.label.mean())
    del usercf_samples
    gc.collect()

    # 召回6: cate pop召回
    cp_samples = pd.read_csv('cate_pop_recall/{}.csv'.format(dtype))
    cp_samples['customer_id'] = cp_samples['customer_id'].map(cust_ex_dic)
    cp_samples['article_id'] = cp_samples['article_id'].map(art_ex_dic)
    cp_samples = cp_samples[cp_samples.customer_id.isin(samples.customer_id.unique())]
    samples = samples.merge(cp_samples, on=['customer_id', 'article_id', 'label'], how='outer')
    # samples = samples.merge(cp_samples[['customer_id', 'article_id', 'label']],
    #                         on=['customer_id', 'article_id', 'label'], how='outer')
    print(samples.shape)
    print('Cate_Pop召回后: ', samples.label.mean())

    samples['popularity'] = samples['article_id'].map(dummy_dict)

    print(list(samples.columns))

    tmp = samples.groupby('customer_id')['label'].agg('sum').reset_index()
    print('召回后具有正样本的用户占比: ', len(tmp[tmp.label > 0]) / len(tmp))

    del data_hist, data_lw, target, data_lm
    gc.collect()

    return samples