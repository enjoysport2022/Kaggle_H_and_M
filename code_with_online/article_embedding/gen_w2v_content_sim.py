import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from gensim.models import Word2Vec
import datetime
from time import time
import warnings

warnings.filterwarnings("ignore")

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


def get_art_sim_dict(df, art_map_dic, topn=100):
    feats = [c for c in df.columns if c not in ['article_id']]
    split_size = 2000
    split_num = int(len(df) / split_size)
    # if len(w2v_df) % split_size != 0:
    if len(df) % split_size != 0:
        split_num += 1

    w2v_vec = df[feats].values

    l2norm = np.linalg.norm(w2v_vec, axis=1, keepdims=True)
    w2v_vec = w2v_vec / (l2norm + 1e-9)

    w2v_vec_T = w2v_vec.T

    art_sim_dict = {}

    cnt = 0

    for i in tqdm(range(split_num)):

        vec = w2v_vec[i * split_size:(i + 1) * split_size]

        sim = vec.dot(w2v_vec_T)

        idx = (-sim).argsort(axis=1)
        sim = (-sim)
        sim.sort(axis=1)

        idx = idx[:, :topn]
        score = sim[:, :topn]
        score = -score

        for idx_, score_ in zip(idx, score):
            idx_ = [art_map_dic[j] for j in idx_]

            art_sim_dict[art_map_dic[cnt]] = dict(zip(idx_, score_))

            cnt += 1

    return art_sim_dict

def generate_w2v_sim(transactions_train, date, last_days = 180, size=5, dtype='offline'):

    save_path = 'sim_dict/w2v/'.format(dtype)

    if not os.path.exists(save_path):

        os.makedirs(save_path)

    w2v_model = get_w2v_model(transactions_train, date, size=size, last_days=last_days, )
    w2v_df = pd.DataFrame()
    w2v_df['article_id'] = w2v_model.wv.index_to_key
    w2v_vectors = pd.DataFrame(w2v_model.wv.vectors,
                               columns=['article_w2v_dim{}'.format(i) for i in range(w2v_model.wv.vector_size)])
    w2v_df = pd.concat([w2v_df, w2v_vectors], axis=1)

    pop_num = 6000
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=7)
    begin_date = str(begin_date).split(' ')[0]
    data_lw = transactions_train[
        (transactions_train.t_dat >= begin_date) & (transactions_train.t_dat <= date)]
    dummy_dict = data_lw['article_id'].value_counts()
    recent_active_articles = list(dummy_dict.index[:pop_num])

    df = w2v_df[w2v_df.article_id.isin(recent_active_articles)]
    art_map_dic = dict(zip(range(len(df)), df['article_id'].values.tolist()))

    art_sim_dict = get_art_sim_dict(df, art_map_dic, topn=200)

    np.save(save_path + 'sim_dict_{}.npy'.format(dtype), art_sim_dict, allow_pickle=True)

articles = pd.read_csv('articles.csv', dtype={'article_id': str})
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})

# date = '2020-09-01'
# recent_active_articles = transactions_train[transactions_train.t_dat >= date]['article_id'].unique()

last_days, size = 180, 32 # 90, 20

generate_w2v_sim(transactions_train, '2020-09-08', last_days=last_days, size=size, dtype='train')

generate_w2v_sim(transactions_train, '2020-09-15', last_days=last_days, size=size, dtype='valid')

generate_w2v_sim(transactions_train, '2020-09-22', last_days=last_days, size=size, dtype='test')


'''
print('offline')
generate_w2v_sim(transactions_train, '2020-09-15', last_days=last_days, size=size, dtype='offline')

print('\n')

print('online')
generate_w2v_sim(transactions_train, '2020-09-22', last_days=last_days, size=size, dtype='online')
'''