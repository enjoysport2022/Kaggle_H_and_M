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
    save_path = 'tfidf_feats/v1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df1 = data[data.t_dat <= date].groupby('customer_id')['article_id'].agg(lambda x: ' '.join(list(x))).reset_index()
    # df1['type'] = 'valid'

    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    df2 = data[data.t_dat <= begin_date].groupby('customer_id')['article_id'].agg(lambda x: ' '.join(list(x))).reset_index()
    # df2['type'] = 'train'

    tfidf_enc = TfidfVectorizer()
    tfidf_enc.fit(df1['article_id'].values.tolist())
    print(len(df1), len(tfidf_enc.vocabulary_), data[data.t_dat <= date]['article_id'].nunique())
    # tfidf_vec = tfidf_enc.fit_transform(df['article_id'].values.tolist())

    tfidf_vec1 = tfidf_enc.transform(df1['article_id'].values.tolist())
    tfidf_vec2 = tfidf_enc.transform(df2['article_id'].values.tolist())

    svd_enc = TruncatedSVD(n_components=size, n_iter=20, random_state=2021)

    svd_enc.fit(tfidf_vec1)

    svd_vec1 = svd_enc.transform(tfidf_vec1)
    svd_vec2 = svd_enc.transform(tfidf_vec2)

    feats = ['article_tfidf_svd_vec_{}'.format(i) for i in range(size)]

    d1 = pd.DataFrame(svd_vec1, columns=feats)
    d_valid = pd.concat([df1[['customer_id']], d1], axis=1)
    d2 = pd.DataFrame(svd_vec2, columns=feats)
    d_train = pd.concat([df2[['customer_id']], d2], axis=1)

    print(d_train.shape)
    print(d_train.head())
    d_train.to_csv(save_path + '{}_train.csv'.format(dtype), index=False)

    print(d_valid.shape)
    print(d_valid.head())
    d_valid.to_csv(save_path + '{}_valid.csv'.format(dtype), index=False)

transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

size = 20
get_tf_idf_feats(transactions_train, '2020-09-15', size=size, dtype='offline')
get_tf_idf_feats(transactions_train, '2020-09-22', size=size, dtype='online')