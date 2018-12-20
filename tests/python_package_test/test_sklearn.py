# coding: utf-8
# pylint: skip-file
import math
import os
import unittest

import lightgbm as lgb
import numpy as np
from sklearn import __version__ as sk_version
from sklearn.base import clone
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris, load_svmlight_file)
from sklearn.exceptions import SkipTestWarning
from sklearn.externals import joblib
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import (_yield_all_checks, SkipTest,
                                            check_parameters_default_constructible)


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


class TestSklearn(unittest.TestCase):

    def test_binary(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMClassifier(n_estimators=50, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
        ret = log_loss(y_test, gbm.predict_proba(X_test))
        self.assertLess(ret, 0.15)
        self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['binary_logloss'][gbm.best_iteration_ - 1], places=5)

    def test_regression(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=50, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
        ret = mean_squared_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 16)
        self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1], places=5)

    def test_multiclass(self):
        X, y = load_digits(10, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMClassifier(n_estimators=50, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
        ret = multi_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.2)
        ret = multi_logloss(y_test, gbm.predict_proba(X_test))
        self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['multi_logloss'][gbm.best_iteration_ - 1], places=5)

    def test_lambdarank(self):
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                           '../../examples/lambdarank/rank.train'))
        X_test, y_test = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                         '../../examples/lambdarank/rank.test'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          '../../examples/lambdarank/rank.train.query'))
        q_test = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         '../../examples/lambdarank/rank.test.query'))
        gbm = lgb.LGBMRanker()
        gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],
                eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=5, verbose=False,
                callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])
        self.assertLessEqual(gbm.best_iteration_, 12)
        self.assertGreater(gbm.best_score_['valid_0']['ndcg@1'], 0.65)
        self.assertGreater(gbm.best_score_['valid_0']['ndcg@3'], 0.65)

    def test_regression_with_custom_objective(self):
        def objective_ls(y_true, y_pred):
            grad = (y_pred - y_true)
            hess = np.ones(len(y_true))
            return grad, hess

        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=50, silent=True, objective=objective_ls)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='regression',
                early_stopping_rounds=5, verbose=False)
        ret = mean_squared_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 100)
        self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1], places=5)

    def test_binary_classification_with_custom_objective(self):
        def logregobj(y_true, y_pred):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            grad = y_pred - y_true
            hess = y_pred * (1.0 - y_pred)
            return grad, hess

        def binary_error(y_test, y_pred):
            return np.mean([int(p > 0.5) != y for y, p in zip(y_test, y_pred)])

        X, y = load_digits(2, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMClassifier(n_estimators=50, silent=True, objective=logregobj)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_error',
                early_stopping_rounds=5, verbose=False)
        ret = binary_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 0.1)

    def test_dart(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(boosting_type='dart')
        gbm.fit(X_train, y_train)
        self.assertLessEqual(gbm.score(X_train, y_train), 1.)

    def test_grid_search(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        params = {'boosting_type': ['dart', 'gbdt'],
                  'n_estimators': [5, 8],
                  'drop_rate': [0.05, 0.1]}
        gbm = GridSearchCV(lgb.LGBMRegressor(), params, cv=3)
        gbm.fit(X_train, y_train)
        self.assertIn(gbm.best_params_['n_estimators'], [5, 8])

    def test_clone_and_property(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=100, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        gbm_clone = clone(gbm)
        self.assertIsInstance(gbm.booster_, lgb.Booster)
        self.assertIsInstance(gbm.feature_importances_, np.ndarray)

        X, y = load_digits(2, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        clf = lgb.LGBMClassifier()
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        self.assertListEqual(sorted(clf.classes_), [0, 1])
        self.assertEqual(clf.n_classes_, 2)
        self.assertIsInstance(clf.booster_, lgb.Booster)
        self.assertIsInstance(clf.feature_importances_, np.ndarray)

    def test_joblib(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=100, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        joblib.dump(gbm, 'lgb.pkl')
        gbm_pickle = joblib.load('lgb.pkl')
        self.assertIsInstance(gbm_pickle.booster_, lgb.Booster)
        self.assertDictEqual(gbm.get_params(), gbm_pickle.get_params())
        self.assertListEqual(list(gbm.feature_importances_), list(gbm_pickle.feature_importances_))

        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        gbm_pickle.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        for key in gbm.evals_result_:
            for evals in zip(gbm.evals_result_[key], gbm_pickle.evals_result_[key]):
                self.assertAlmostEqual(*evals, places=5)
        pred_origin = gbm.predict(X_test)
        pred_pickle = gbm_pickle.predict(X_test)
        self.assertEqual(len(pred_origin), len(pred_pickle))
        for preds in zip(pred_origin, pred_pickle):
            self.assertAlmostEqual(*preds, places=5)

    def test_feature_importances_single_leaf(self):
        clf = lgb.LGBMClassifier(n_estimators=100)
        data = load_iris()
        clf.fit(data.data, data.target)
        importances = clf.feature_importances_
        self.assertEqual(len(importances), 4)

    def test_feature_importances_type(self):
        clf = lgb.LGBMClassifier(n_estimators=100)
        data = load_iris()
        clf.fit(data.data, data.target)
        clf.set_params(importance_type='split')
        importances_split = clf.feature_importances_
        clf.set_params(importance_type='gain')
        importances_gain = clf.feature_importances_
        # Test that the largest element is NOT the same, the smallest can be the same, i.e. zero
        importance_split_top1 = sorted(importances_split, reverse=True)[0]
        importance_gain_top1 = sorted(importances_gain, reverse=True)[0]
        self.assertNotEqual(importance_split_top1, importance_gain_top1)

    # sklearn <0.19 cannot accept instance, but many tests could be passed only with min_data=1 and min_data_in_bin=1
    @unittest.skipIf(sk_version < '0.19.0', 'scikit-learn version is less than 0.19')
    def test_sklearn_integration(self):
        # we cannot use `check_estimator` directly since there is no skip test mechanism
        for name, estimator in ((lgb.sklearn.LGBMClassifier.__name__, lgb.sklearn.LGBMClassifier),
                                (lgb.sklearn.LGBMRegressor.__name__, lgb.sklearn.LGBMRegressor)):
            check_parameters_default_constructible(name, estimator)
            # we cannot leave default params (see https://github.com/Microsoft/LightGBM/issues/833)
            estimator = estimator(min_child_samples=1, min_data_in_bin=1)
            for check in _yield_all_checks(name, estimator):
                check_name = check.func.__name__ if hasattr(check, 'func') else check.__name__
                if check_name == 'check_estimators_nan_inf':
                    continue  # skip test because LightGBM deals with nan
                try:
                    check(name, estimator)
                except SkipTest as message:
                    warnings.warn(message, SkipTestWarning)

    @unittest.skipIf(not lgb.compat.PANDAS_INSTALLED, 'pandas is not installed')
    def test_pandas_categorical(self):
        import pandas as pd
        X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                          "B": np.random.permutation([1, 2, 3] * 100),  # int
                          "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                          "D": np.random.permutation([True, False] * 150)})  # bool
        y = np.random.permutation([0, 1] * 150)
        X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),
                               "B": np.random.permutation([1, 3] * 30),
                               "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                               "D": np.random.permutation([True, False] * 30)})
        for col in ["A", "B", "C", "D"]:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        gbm0 = lgb.sklearn.LGBMClassifier().fit(X, y)
        pred0 = list(gbm0.predict(X_test))
        gbm1 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=[0])
        pred1 = list(gbm1.predict(X_test))
        gbm2 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=['A'])
        pred2 = list(gbm2.predict(X_test))
        gbm3 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
        pred3 = list(gbm3.predict(X_test))
        gbm3.booster_.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = list(gbm4.predict(X_test))
        pred_prob = list(gbm0.predict_proba(X_test)[:, 1])
        np.testing.assert_almost_equal(pred0, pred1)
        np.testing.assert_almost_equal(pred0, pred2)
        np.testing.assert_almost_equal(pred0, pred3)
        np.testing.assert_almost_equal(pred_prob, pred4)

    def test_predict(self):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                            test_size=0.2, random_state=42)

        gbm = lgb.train({'objective': 'multiclass',
                         'num_class': 3,
                         'verbose': -1},
                        lgb.Dataset(X_train, y_train))
        clf = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train)

        # Tests same probabilities
        res_engine = gbm.predict(X_test)
        res_sklearn = clf.predict_proba(X_test)
        np.testing.assert_allclose(res_engine, res_sklearn)

        # Tests same predictions
        res_engine = np.argmax(gbm.predict(X_test), axis=1)
        res_sklearn = clf.predict(X_test)
        np.testing.assert_equal(res_engine, res_sklearn)

        # Tests same raw scores
        res_engine = gbm.predict(X_test, raw_score=True)
        res_sklearn = clf.predict(X_test, raw_score=True)
        np.testing.assert_allclose(res_engine, res_sklearn)

        # Tests same leaf indices
        res_engine = gbm.predict(X_test, pred_leaf=True)
        res_sklearn = clf.predict(X_test, pred_leaf=True)
        np.testing.assert_equal(res_engine, res_sklearn)

        # Tests same feature contributions
        res_engine = gbm.predict(X_test, pred_contrib=True)
        res_sklearn = clf.predict(X_test, pred_contrib=True)
        np.testing.assert_allclose(res_engine, res_sklearn)

        # Tests other parameters for the prediction works
        res_engine = gbm.predict(X_test)
        res_sklearn_params = clf.predict_proba(X_test,
                                               pred_early_stop=True,
                                               pred_early_stop_margin=1.0)
        self.assertRaises(AssertionError,
                          np.testing.assert_allclose,
                          res_engine, res_sklearn_params)

    def test_evaluate_train_set(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=10, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        self.assertIn('training', gbm.evals_result_)
        self.assertIn('l2', gbm.evals_result_['training'])
        self.assertIn('valid_1', gbm.evals_result_)
        self.assertIn('l2', gbm.evals_result_['valid_1'])

    def test_regressor_train_and_valid_metric(self):
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMRegressor(n_estimators=10, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_train_metric='l2', eval_valid_metric=['l1', 'l2'],
                early_stopping_rounds=5, verbose=False)
        self.assertIn('l2', gbm.evals_result_['training'])
        self.assertIn('l1', gbm.evals_result_['valid_1'])
        self.assertIn('l2', gbm.evals_result_['valid_1'])

    def test_classifier_train_and_valid_metric(self):
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        gbm = lgb.LGBMClassifier(n_estimators=10, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_train_metric='binary_error', eval_valid_metric=['binary_logloss', 'binary_error'],
                early_stopping_rounds=5, verbose=False)
        self.assertIn('binary_error', gbm.evals_result_['training'])
        self.assertIn('binary_error', gbm.evals_result_['valid_1'])
        self.assertIn('binary_logloss', gbm.evals_result_['valid_1'])

    def test_ranker_train_and_valid_metric(self):
        X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                           '../../examples/lambdarank/rank.train'))
        X_test, y_test = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                         '../../examples/lambdarank/rank.test'))
        q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          '../../examples/lambdarank/rank.train.query'))
        q_test = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         '../../examples/lambdarank/rank.test.query'))
        gbm = lgb.LGBMRanker(n_estimators=10, silent=True)
        gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_group=[q_train, q_test], eval_at=[1, 3],
                eval_train_metric='ndcg', eval_valid_metric=['l2', 'ndcg'],
                early_stopping_rounds=5, verbose=False)
        self.assertIn('ndcg@1', gbm.evals_result_['training'])
        self.assertIn('ndcg@3', gbm.evals_result_['training'])
        self.assertIn('ndcg@1', gbm.evals_result_['valid_1'])
        self.assertIn('ndcg@3', gbm.evals_result_['valid_1'])
        self.assertIn('l2', gbm.evals_result_['valid_1'])
