import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import gc
from gensim.models import Word2Vec
from time import time
import datetime
import warnings
warnings.filterwarnings('ignore')


def train_model(data, size=10, save_path='w2v_model/', iter=5, window=20):
    """训练模型"""
    print('Begin training w2v model')
    begin_time = time()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = Word2Vec(data, vector_size=size, window=window, min_count=0, workers=20,
                     seed=1997, epochs=iter, sg=1, hs=1, compute_loss=True,
                     # min_alpha=0.005
                     )
    print(model.get_latest_training_loss())

    # model.save(save_path + 'all_click.model')
    model.save(save_path + 'w2v.model')

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', round(run_time, 2))  # 该循环程序运行时间： 1.4201874732
    return model


def get_w2v_model(df_, date, last_days=30, size=10, iter=5, save_path='w2v_model/', window=20, new_vocab=False,
                  day_split=False):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)

    begin_date = str(begin_date).split(' ')[0]

    df = df_[(df_.t_dat <= date) & (df_.t_dat >= begin_date)]

    # df['article_id'] = df['article_id'].astype(str)

    user_item = df.groupby('customer_id')['article_id'].agg(list).reset_index()
    model = train_model(user_item['article_id'].values, size=size, iter=iter, save_path=save_path, window=window)

    # item_emb, emb_cols = train_model(user_item['item_id'].values, size=10)

    return model

def generate_w2v_feats(transactions_train, date, last_days = 180, size=5, dtype='offline'):

    save_path = 'w2v_feats/'

    if not os.path.exists(save_path):

        os.makedirs(save_path)

    w2v_model = get_w2v_model(transactions_train, date, size=size, last_days=last_days, )
    article_w2v_df = pd.DataFrame()
    article_w2v_df['article_id'] = w2v_model.wv.index_to_key
    w2v_vectors = pd.DataFrame(w2v_model.wv.vectors,
                               columns=['article_w2v_dim{}'.format(i) for i in range(w2v_model.wv.vector_size)])
    article_w2v_df = pd.concat([article_w2v_df, w2v_vectors], axis=1)

    print(article_w2v_df.shape)
    print(article_w2v_df.head())

    article_w2v_df.to_csv(save_path + '{}_w2v_feats.csv'.format(dtype), index=False)
    # return article_w2v_df

transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])

last_days, size = 90, 20 # 90, 20

print('offline')
generate_w2v_feats(transactions_train, '2020-09-15', last_days=last_days, size=size, dtype='offline')

print('\n')

print('online')
generate_w2v_feats(transactions_train, '2020-09-22', last_days=last_days, size=size, dtype='online')