Thanks my teammate @Ryan, we got 16th place.
Thanks to H&M and Kaggle. This is a wonderful competition.

# Framework
![Framework](https://tva1.sinaimg.cn/large/e6c9d24ely1h23j5ugbfnj21fn0u0wid.jpg)

# Data-splitting
We split the data into 3 groups: 
cg1(customer group 1) is the users with transactions in last 30 days; 
cg2(customer group 2) is the users without transactions in last 30 days, 
but with transactions in history. 
cg3(customer group 3) is the users without transactions in history. 

- cg1 and cg2: multi-recalls + rank.
- cg3: popular items recall.

# Recalls
- popular items recall
- repurchase recall
- binaryNet recall
- ItemCF recall
- UserCF recall
- W2V content recall
- NLP content recall
- Image content recall
- Category content recall

Each recall method will recall 100 items for every user,
then drop duplicates.

# Rank

## Feature Engineer
### Item Feature
groupby article_id agg cols calculate statistics
- cols: customer_id, price, sales_channel_id, and so on.
- op: 'min', 'max', 'mean', 'std', 'median', 'sum', 'nunique'

### User Feature
groupby customer_id agg cols calculate statistics
- cols: price, article_id, sales_channel_id, and so on.
- op: 'min', 'max', 'mean', 'std', 'median', 'sum', 'nunique'

### Interaction Feature
- count of user-item purchased in different window(1day, 3days, 1week, 2weeks, 1month).
- The time-diff since the user last purchased the item

### other features
- ItemCF score
- BinaryNet score

### Model
- lightgbm ranker
- lightgbm binary

# Ensemble
Ref to [this link](https://www.kaggle.com/code/tarique7/lb-0-0240-h-m-ensemble-magic-multi-blend)
