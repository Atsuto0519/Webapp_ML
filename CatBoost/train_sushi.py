import pickle
import pandas as pd
from catboost import *
from sklearn.model_selection import train_test_split

train_path = "./data/sushi3-2016/train.csv"
model_path = "./model.pkl"

df = pd.read_table(train_path,sep=' ',header=None)
df = df.rename(columns={0: 'user_id', 19: 'score'})

# テストデータには，最後のuser_idの人のみを使用．
train_df = df[0:len(df)-100]
test_df = df[len(df)-100:len(df)]


cat_features = list(range(12))

train_data = train_df.drop(["user_id", "score"], axis=1)
train_label = train_df["score"]
train_group_id = train_df["user_id"]
train_pool = Pool(data=train_data, label=train_label, group_id=train_group_id, cat_features=cat_features)

test_data = test_df.drop(["user_id", "score"], axis=1)
test_label = test_df["score"]
test_group_id = test_df["user_id"]
test_pool = Pool(data=test_data, label=test_label, group_id=test_group_id, cat_features=cat_features)

# YetiRankはpairwiseでのランク学習の1つ
param = {'loss_function':'YetiRank', 'learning_rate': .05, 'iterations': 200,
         'depth': 4, 'use_best_model':True}
model = CatBoost(param)
model.fit(train_pool, eval_set=test_pool)

with open(model_path, mode='wb') as f:
    pickle.dump(model, f)
