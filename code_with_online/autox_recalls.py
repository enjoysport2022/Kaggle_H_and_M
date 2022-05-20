import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import gc

def popular_recall(uids, data, date, uid, iid, time_col, last_days=7, recall_num=100, dtype='train'):

    assert dtype in ['train', 'test']

    if dtype == 'train':
        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date).split(' ')[0]
        pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        pop_begin_date = str(pop_begin_date).split(' ')[0]

        target = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target[time_col].min(), target[time_col].max())
        target = target.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        data_lw = data[(data[time_col] >= pop_begin_date) & (data[time_col] <= begin_date)]
        popular_item = list(data_lw[iid].value_counts().index[:recall_num])

        samples = []
        hit = 0
        for cur_uid, labels in tqdm(target.values):
            h = 0
            for cur_iid in popular_item:
                if cur_iid in labels:
                    sample = [cur_uid, cur_iid, 1]
                    h += 1
                else:
                    sample = [cur_uid, cur_iid, 0]
                samples.append(sample)
            hit += h / len(labels)
        print('HIT: ', hit / len(target))
        samples = pd.DataFrame(samples, columns=[uid, iid, 'label'])

        return samples

    elif dtype == 'test':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date).split(' ')[0]

        data_lw = data[(data[time_col] >= begin_date) & (data[time_col] <= date)]
        popular_item = list(data_lw[iid].value_counts().index[:recall_num])

        samples = []
        for cur_uid in tqdm(uids):
            for cur_iid in popular_item:
                samples.append([cur_uid, cur_iid])
        samples = pd.DataFrame(samples, columns=[uid, iid])

        return samples


def history_recall(uids, data, date, uid, iid, time_col, last_days=7, recall_num=100, dtype='train'):
    assert dtype in ['train', 'test']

    if dtype == 'train':
        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date).split(' ')[0]

        data_hist = data[data.t_dat <= begin_date]
        
        target = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target[time_col].min(), target[time_col].max())
        target = target.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        purchase_df = data_hist[data_hist[uid].isin(target[uid].unique())].groupby(uid).tail(recall_num).reset_index(drop=True)
        purchase_df = purchase_df[[uid, iid]]
        purchase_df = purchase_df.groupby(uid)[iid].agg(list).reset_index()
        purchase_df = purchase_df.merge(target, on=uid, how='left')

        samples = []
        for cur_uid, cur_iids, label in tqdm(purchase_df.values, total=len(purchase_df)):
            for cur_iid in cur_iids:
                if cur_iid in label:
                    samples.append([cur_uid, cur_iid, 1])
                else:
                    samples.append([cur_uid, cur_iid, 0])

        samples = pd.DataFrame(samples, columns=[uid, iid, 'label'])

        return samples

    elif dtype == 'test':

        purchase_df = data.loc[data[uid].isin(uids)].groupby(uid).tail(recall_num).reset_index(drop=True)
        samples = purchase_df[[uid, iid]]

        return samples

