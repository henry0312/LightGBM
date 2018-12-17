import numpy
import lightgbm
from lightgbm.sklearn import LGBMRegressor, LGBMModel, LGBMRanker
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


X, y = make_regression(n_samples=100, n_features=10, random_state=42)
scaler = MinMaxScaler((0, 20))
y = scaler.fit_transform(y.reshape(-1, 1))
y = y.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test = y_test.astype(int)  # label should be int type
group_train = [10 for _ in range(7)]
group_test = [10 for _ in range(3)]

# gbm = LGBMModel(
#         boosting_type='gbdt',
#         num_leaves=15,
#         max_depth=-1,
#         learning_rate=0.01,
#         n_estimators=1000,
#         objective='regression',
#         random_state=42)
# gbm.fit(X_train, y_train,
#         eval_metric='ndcg', eval_set=[(X_test, y_test)], eval_group=[group_test],
#         early_stopping_rounds=10)

# gbm = LGBMRanker(
#         boosting_type='gbdt',
#         num_leaves=15,
#         max_depth=-1,
#         learning_rate=0.01,
#         n_estimators=1000,
#         objective=None,
#         label_gain=list(range(0, numpy.max(y_test) + 1)),
#         random_state=42)
# gbm.fit(X_train, y_train, group=group_train,
#         eval_metric='ndcg', eval_set=[(X_test, y_test)], eval_group=[group_test],
#         early_stopping_rounds=10)


# create dataset for lightgbm
lgb_train = lightgbm.Dataset(X_train, y_train, group=group_train)
lgb_eval = lightgbm.Dataset(X_test, y_test, group=group_test)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['ndcg'],
    'eval_at': [5],
    'label_gain': list(range(0, numpy.max(y_test) + 1)),
    'num_leaves': 15,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
}

gbm = lightgbm.train(params,
                     lgb_train,
                     num_boost_round=1000,
                     valid_sets=[lgb_eval],
                     early_stopping_rounds=10)
