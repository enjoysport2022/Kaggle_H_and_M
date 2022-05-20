import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from time import time
import gc
import datetime
import warnings

warnings.filterwarnings("ignore")


def get_tf_idf_feats(data, date, size=5, last_days=7, dtype='offline'):
    save_path = 'tfidf_feats/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df1 = data[data.t_dat <= date].groupby('customer_id')['article_id'].agg(lambda x: ' '.join(list(x))).reset_index()
    df1['type'] = 'valid'

    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    df2 = data[data.t_dat <= begin_date].groupby('customer_id')['article_id'].agg(lambda x: ' '.join(list(x))).reset_index()
    df2['type'] = 'train'

    df = pd.concat([df2, df1]).reset_index(drop=True)

    tfidf_enc = TfidfVectorizer()

    tfidf_vec = tfidf_enc.fit_transform(df['article_id'].values.tolist())

    svd_enc = TruncatedSVD(n_components=size, n_iter=5, random_state=2021)

    svd_vec = svd_enc.fit_transform(tfidf_vec)

    d = pd.DataFrame(svd_vec)

    d.columns = ['article_svd_vec_{}'.format(i) for i in range(size)]

    d = pd.concat([d, df[['customer_id', 'type']]], axis=1)

    del df

    gc.collect()

    d_train = d[d['type'] == 'train']
    d_train.drop(columns=['type'])
    print(d_train.shape)
    print(d_train.head())
    d_train.to_csv(save_path + '{}_train.csv'.format(dtype), index=False)

    d_valid = d[d['type'] == 'valid']
    d_valid.drop(columns=['type'])
    print(d_valid.shape)
    print(d_valid.head())
    d_valid.to_csv(save_path + '{}_valid.csv'.format(dtype), index=False)

transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

get_tf_idf_feats(transactions_train, '2020-09-15', size=10, dtype='offline')
get_tf_idf_feats(transactions_train, '2020-09-22', size=10, dtype='online')