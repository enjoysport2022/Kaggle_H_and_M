from XX import user_feature_engineer, item_feature_engineer
from XX import ranker
from XX import process_recall

class AutoXRecommend():
    def __init__(self):
        pass

    def recall(self, uids, method):
        if method == 'popular':
            return []

        elif method == 'itemcf':
            return []

    def fit(self, interaction, user_info, item_info, uid, iid, mode='recall_and_rank', recall_method=None):

        assert mode in ['recall', 'recall_and_rank']

        if mode == 'recall':
            assert recall_method in ['popular', 'itemcf']
            recall = self.recall(interaction[uid].unique, method=recall_method, dtype='test')
            result = process_recall(recall)
            return result

        # 召回
        # popular_recall
        print('popular_recall, train')
        popular_recall_train = popular_recall(None, transactions_train, date='2020-09-15',
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')

        print('popular_recall, valid')
        popular_recall_valid = popular_recall(None, transactions_train, date='2020-09-22',
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')

        print('history_recall, train')
        history_recall_train = history_recall(None, transactions_train, date='2020-09-15',
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')
        print('history_recall, valid')
        history_recall_valid = history_recall(None, transactions_train, date='2020-09-22',
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')

        print('itemcf_recall, train')
        itemcf_recall_train = itemcf_recall(None, transactions_train, date='2020-09-15',
                      uid=uid, iid=iid, time_col=time_col,
                      last_days=7, recall_num=100, dtype='train',
                      topk=1000, use_iif=False, sim_last_days=14)

        print('itemcf_recall, valid')
        itemcf_recall_valid = itemcf_recall(None, transactions_train, date='2020-09-22',
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='train',
                                            topk=1000, use_iif=False, sim_last_days=14)

        print('binary_recall, train')
        binary_recall_train = binary_recall(None, transactions_train, date='2020-09-15',
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='train', topk=1000)

        print('binary_recall, valid')
        binary_recall_valid = binary_recall(None, transactions_train, date='2020-09-22',
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='train', topk=1000)

        # 合并数据
        train = popular_recall_train.append(history_recall_train)
        train.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)

        train = train.merge(itemcf_recall_train, on=['customer_id', 'article_id', 'label'], how='outer')
        train = train.merge(binary_recall_train, on=['customer_id', 'article_id', 'label'], how='outer')


        valid = popular_recall_valid.append(history_recall_valid)
        valid.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)

        valid = valid.merge(itemcf_recall_valid, on=['customer_id', 'article_id', 'label'], how='outer')
        valid = valid.merge(binary_recall_valid, on=['customer_id', 'article_id', 'label'], how='outer')


        # 特征构造
        train = feature_engineer(train, transactions_train,
                                 date='2020-09-15',
                                 customers=customers, articles=articles,
                                 uid=uid, iid=iid, time_col=time_col,
                                 last_days=7, dtype='train')
        valid = feature_engineer(valid, transactions_train,
                                 date='2020-09-22',
                                 customers=customers, articles=articles,
                                 uid=uid, iid=iid, time_col=time_col,
                                 last_days=7, dtype='train')

        # 排序
        lgb_ranker, valid = ranker(train, valid,
                                   uid=uid, iid=iid, time_col=time_col)



        # 重新运行
        # 召回
        train_date = '2020-09-22'
        test_date = '2020-09-22'
        all_user = transactions_train[uid].unique()
        popular_recall_train = popular_recall(None, transactions_train, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')
        popular_recall_test = popular_recall(all_user, transactions_train, date=test_date,
                                             uid=uid, iid=iid, time_col=time_col,
                                             last_days=7, recall_num=100, dtype='test')

        history_recall_train = history_recall(None, transactions_train, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=100, dtype='train')
        history_recall_test = history_recall(all_user, transactions_train, date=test_date,
                                             uid=uid, iid=iid, time_col=time_col,
                                             last_days=7, recall_num=100, dtype='test')

        itemcf_recall_train = itemcf_recall(None, transactions_train, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='train',
                                            topk=1000, use_iif=False, sim_last_days=14)
        itemcf_recall_test = itemcf_recall(all_user, transactions_train, date=test_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='test',
                                            topk=1000, use_iif=False, sim_last_days=14)

        binary_recall_train = binary_recall(None, transactions_train, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='train', topk=1000)
        binary_recall_test = binary_recall(all_user, transactions_train, date=test_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=100, dtype='test', topk=1000)

        # 合并数据
        train = popular_recall_train.append(history_recall_train)
        train.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)
        train = train.merge(itemcf_recall_train, on=['customer_id', 'article_id', 'label'], how='outer')
        train = train.merge(binary_recall_train, on=['customer_id', 'article_id', 'label'], how='outer')

        test = popular_recall_test.append(history_recall_test)
        test.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)
        test = test.merge(itemcf_recall_test, on=['customer_id', 'article_id'], how='outer')
        test = test.merge(binary_recall_test, on=['customer_id', 'article_id'], how='outer')

        # 特征构造
        train = feature_engineer(train, transactions_train,
                                 date='2020-09-22',
                                 customers=customers, articles=articles,
                                 uid=uid, iid=iid, time_col=time_col,
                                 last_days=7, dtype='train')

        test = feature_engineer(test, transactions_train,
                                date='2020-09-22',
                                customers=customers, articles=articles,
                                uid=uid, iid=iid, time_col=time_col,
                                last_days=7, dtype='test')

        # 排序
        ranker(train, test)


    def transform(self, uids):

        # 召回
        popular_recall_test = self.recall_test(uids, method='popular')
        itemcf_recall_test = self.recall_test(uids, method='itemcf')

        test = popular_recall_test.append(itemcf_recall_test)
        test.drop_duplicates(subset=[self.uid, self.iid], keep='first', inplace=True)

        # 特征构造
        test = test.merge(self.user_fe, on=self.uid, how='left')
        test = test.merge(self.item_fe, on=self.iid, how='left')
        test = test.merge(self.inter_fe, on=[self.iid, self.iid], how='left')

        # 排序
        result = ranker.precict(test)
        return result
