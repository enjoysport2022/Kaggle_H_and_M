import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from gensim.models import Word2Vec
from time import time
import warnings

warnings.filterwarnings("ignore")



articles = pd.read_csv('articles.csv', dtype={'article_id': str})
transactions_train = pd.read_csv('transactions_train.csv', dtype={'article_id': str})

size = 128
tfidf_enc = TfidfVectorizer()
tfidf_vec = tfidf_enc.fit_transform(articles['detail_desc'].fillna('').values.tolist())
svd_enc = TruncatedSVD(n_components=size, n_iter=20, random_state=2021)
svd_vec = svd_enc.fit_transform(tfidf_vec)
df = pd.DataFrame(svd_vec, columns=['detail_tfidf_vec{}'.format(i) for i in range(size)])
df['article_id'] = articles['article_id']

# date = '2020-09-01'
# recent_active_articles = transactions_train[transactions_train.t_dat >= date]['article_id'].unique()
pop_num = 6000 # 1000
data_lw = transactions_train[(transactions_train.t_dat >= '2020-09-07') & (transactions_train.t_dat <= '2020-09-15')]
dummy_dict = data_lw['article_id'].value_counts()
recent_active_articles = list(dummy_dict.index[:pop_num])
df = df[df.article_id.isin(recent_active_articles)]
art_map_dic = dict(zip(range(df.shape[0]), df['article_id'].values.tolist()))
print(df.shape)

svd_vec = df[['detail_tfidf_vec{}'.format(i) for i in range(size)]].values
l2norm1 = np.linalg.norm(svd_vec,axis=1,keepdims=True)
svd_vec = svd_vec /(l2norm1+1e-9)
split_size = 2000
split_num = int(svd_vec.shape[0]/split_size)
if svd_vec.shape[0]%split_size != 0:
    split_num += 1
svd_vec_T = svd_vec.T


def get_art_sim_dict(split_num, topn=100):
    art_detail_sim_dict = {}

    cnt = 0

    for i in tqdm(range(split_num)):

        vec = svd_vec[i * split_size:(i + 1) * split_size]

        sim = vec.dot(svd_vec_T)

        idx = (-sim).argsort(axis=1)
        sim = (-sim)
        sim.sort(axis=1)

        idx = idx[:, :topn]
        score = sim[:, :topn]
        score = -score

        for idx_, score_ in zip(idx, score):
            idx_ = [art_map_dic[j] for j in idx_]

            art_detail_sim_dict[art_map_dic[cnt]] = dict(zip(idx_, score_))

            cnt += 1

    return art_detail_sim_dict
art_detail_sim_dict = get_art_sim_dict(split_num, topn=300)

save_path = 'sim_dict/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(save_path + 'art_detail_sim_dict_v1.npy', art_detail_sim_dict, allow_pickle=True)