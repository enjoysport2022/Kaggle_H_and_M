import lightgbm as lgb
import pandas as pd

def ranker(train, valid):
    train.sort_values(by=['customer_id'], inplace=True)

    g_train = train.groupby(['customer_id'], as_index=False).count()["label"].values

    valid.sort_values(by=['customer_id'], inplace=True)

    g_val = valid.groupby(['customer_id'], as_index=False).count()["label"].values

    del_cols = []  # + list(tfidf_df.columns)[:5]# ['popularity', 'popularity_last_1days', 'popularity_last_month']

    feats = [f for f in train.columns if f not in ['customer_id', 'article_id', 'label', 'prob'] + del_cols]

    # cate_feats = [c for c in list(customers.columns) + list(articles.columns) if c not in ['customer_id', 'article_id']]
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1)

    lgb_ranker.fit(train[feats], train['label'], group=g_train,
                   eval_set=[(valid[feats], valid['label'])],
                   eval_group=[g_val], eval_at=[12], eval_metric=['map', ],
                   early_stopping_rounds=100, # 50,
                   verbose=50,
                   # categorical_feature=cate_feats
                   )

    print(lgb_ranker.best_score_)

    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = lgb_ranker.feature_importances_

    print(importance_df.sort_values('importance', ascending=False).head(20))

    valid['prob'] = lgb_ranker.predict(valid[feats], num_iteration=lgb_ranker.best_iteration_)

    return lgb_ranker, valid
