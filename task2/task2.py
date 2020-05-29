import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import argparse
import warnings
import tqdm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from score_submission import score_submission
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone
from sklearn.utils import class_weight
from sklearn import linear_model
from sklearn import utils
from sklearn.feature_selection import SelectFromModel


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)
# =============Add different models here!!!!=============
model_heads = []
models = []
from sklearn import tree  # 0

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model_heads.append("Decision Tree Regression\t\t")
models.append(model_DecisionTreeRegressor)

from sklearn import linear_model  # 1

model_LinearRegression = linear_model.LinearRegression()
model_heads.append("Linear Regression\t\t\t\t")
models.append(model_LinearRegression)

from sklearn import svm  # 2

model_SVR = svm.SVR()
model_heads.append("Support Vector Machine Regression")
models.append(model_SVR)

from sklearn import neighbors  # 3

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
model_heads.append("K-Nearest Neighbor Regression\t")
models.append(model_KNeighborsRegressor)

from sklearn import ensemble  # 4

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
model_heads.append("Random Forest Regression\t\t")
models.append(model_RandomForestRegressor)

from sklearn import ensemble  # 5

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=150)
model_heads.append("AdaBoost Regression\t\t\t\t")
models.append(model_AdaBoostRegressor)

from sklearn import ensemble  # 6

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor()
model_heads.append("Gradient Boosting Regression\t")
models.append(model_GradientBoostingRegressor)

from sklearn.ensemble import BaggingRegressor  # 7

model_BaggingRegressor = BaggingRegressor()
model_heads.append("Bagging Regression\t\t\t\t")
models.append(model_BaggingRegressor)

from sklearn.tree import ExtraTreeRegressor  # 8

model_ExtraTreeRegressor = ExtraTreeRegressor()
model_heads.append("ExtraTree Regression\t\t\t")
models.append(model_ExtraTreeRegressor)

# params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
#           'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
model_XGBoostRegressor = xgb.XGBRegressor()
model_heads.append("XGBoost Regression\t\t\t\t")
models.append(model_XGBoostRegressor)
# =============Model Adding Ends=============

# =============For Esemble and Stacking =============
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                            max_depth=4, max_features='sqrt',
                                            min_samples_leaf=15, min_samples_split=10,
                                            loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


def get_model_score(model, x_all, y_all, n_folds=5, mode='esm'):
    kf = KFold(n_splits=n_folds, shuffle=True)
    score_mean_test = 0
    score_mean_train = 0
    for train_idx, test_idx in kf.split(x_all):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]
        y_test = y_all[test_idx]
        score_test, score_train = try_different_method(model, x_train, y_train, x_test, y_test, mode)
        score_mean_test += score_test
        score_mean_train += score_train
    score_mean_test /= n_folds
    score_mean_train /= n_folds
    return score_mean_test


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    """

    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    """

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# =============For Esemble and Stacking(end)=============


def load_data(x_path='./X_train.csv', y_path='./y_train.csv', x_test_path='./X_test.csv'):
    """
    Load data from .csv files
    :param x_path: relative path of x
    :param y_path: relative path of y
    :param x_test_path :relative path of x_test
    :return data_x, data_y, data_x_test: X, Y, X_test in pd.DataFrame format
    """
    print()
    print("Loading data from {}, {} and {}".format(x_path, y_path, x_test_path))
    data_x = pd.read_csv(x_path)
    data_y = pd.read_csv(y_path)
    data_x_test = pd.read_csv(x_test_path)
    print('Data loaded, data_set Information:')
    print("x: {}".format(data_x.shape))
    print("y: {}".format(data_y.shape))
    print("x_test: {}".format(data_x_test.shape))
    # print(range(0, data_y.columns.shape[0]-1))
    return data_x, data_y, data_x_test


def from_csv_to_ndarray(data):
    """
    Fransfer data from pd.DataFrame to ndarray for later model training
    :param data: data in pd.DataFrame
    :return ndarray: data in ndarray
    """
    data.head()
    ndarray = data.values
    if ndarray.shape[1] == 2:
        return ndarray[:, 1]
    else:
        return ndarray[:, 1:]


def fill_missing_data(data_x):
    """
    Fill Nan value in data of pd.DataFrame format
    :param data_x: feature or data in pd.DataFrame format
    :return data_x_filled: filled data
    """
    print("Filling missing data...")
    data_x_filled = data_x.fillna(data_x.median())
    print("Filling missing data completed.")
    return data_x_filled


def data_preprocessing(x_raw, x_test_raw):
    """
    Data preprocessing including normalization or scaling, etc.
    :param x_raw: np.ndarray, data before preprocessing
    :param x_test_raw: np.ndarray, test data before preprocessing
    :return x_after, x_test_after: data after preprocessing
    """
    std = StandardScaler()
    x_concat = np.concatenate((x_raw, x_test_raw), axis=0)
    x_after = std.fit_transform(x_concat)
    return x_after[:len(x_raw)], x_after[len(x_raw):]


def over_sampling(x_train, y_train):
    print()
    print("Doing over sampling...")
    print("Before over sampling:")
    class0_num = np.sum(y_train == 0)
    class1_num = np.sum(y_train == 1)
    class2_num = np.sum(y_train == 2)
    print("#Sample in Class 0: {}".format(class0_num))
    print("#Sample in Class 1: {}".format(class1_num))
    print("#Sample in Class 2: {}".format(class2_num))
    x_out = x_train
    y_out = y_train

    print("After over sampling:")
    class0_num = np.sum(y_out == 0)
    class1_num = np.sum(y_out == 1)
    class2_num = np.sum(y_out == 2)
    print("#Sample in Class 0: {}".format(class0_num))
    print("#Sample in Class 1: {}".format(class1_num))
    print("#Sample in Class 2: {}".format(class2_num))

    return x_out, y_out


def select_feature(x_train, y_train, x_test):
    """
    Select features based on training data(but actually we can use all data)
    :param x_train: features
    :param y_train: labels
    :param x_test: test features
    :return x_selected, y_train, x_test_selected: return selected feature and label
    """
    print()
    print("Selecting features...")
    print("Before feature selection: {}".format(x_train.shape))
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight="balanced")
    random_feature = np.random.rand(x_train.shape[0] * 5).reshape((x_train.shape[0], 5))
    x_train_new = np.concatenate((x_train, random_feature), axis=1)
    lab_enc = preprocessing.LabelEncoder()
    y_train = lab_enc.fit_transform(y_train)
    clf.fit(x_train_new, y_train)
    random_feature_importance = clf.feature_importances_[-5:]
    feature_importance = clf.feature_importances_[:-5]
    for i in range(5):
        feature_importance[feature_importance < random_feature_importance[i]] = 0
    select_idx = feature_importance > 0
    x_selected = x_train[:, select_idx]
    x_test_selected = x_test[:, select_idx]
    print("After feature selection: {}".format(x_selected.shape))
    print("Selecting features completed.")
    return x_selected, y_train, x_test_selected


def try_different_method(model, x_train, y_train, x_test, y_test, mode):
    """
    Inner function in train_evaluate_return_best_model for model training.
    :param model: one specific model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return score:
    """
    model.fit(x_train, y_train)
    result_test = model.predict(x_test)
    result_train = model.predict(x_train)
    if mode == 'clf':
        score_test = roc_auc_score(result_test, y_test)
        score_train = roc_auc_score(result_train, y_train)
    elif mode == 'reg':
        score_test = 0.5 + 0.5 * np.maximum(0, r2_score(result_test, y_test))
        score_train = 0.5 + 0.5 * np.maximum(0, r2_score(result_train, y_train))
        # score_test = r2_score(result_test, y_test)
        # score_train = r2_score(result_train, y_train)
    else:
        score_test = r2_score(result_test, y_test)
        score_train = r2_score(result_train, y_train)
    return score_test, score_train


def evaluate_model(model, x_all, y_all, mode):
    print("Evaluating model...")
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True)
    score_mean_val = 0
    score_mean_train = 0
    for train_idx, val_idx in kf.split(x_all):
        instance_model = clone(model)
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        # print("After kFold split, x_train: {}".format(x_train.shape))
        # print("After kFold split, y_train: {}".format(y_train.shape))
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]
        score_val, score_train = try_different_method(instance_model, x_train, y_train, x_val, y_val, mode)
        score_mean_val += score_val
        score_mean_train += score_train

    score_mean_val /= n_folds
    score_mean_train /= n_folds
    print("Mean score on val set: {}".format(score_mean_val))
    print("Mean score on train set: {}".format(score_mean_train))
    model.fit(x_all, y_all)
    return model


def main():
    print()
    print('***************By Manyeo***************')
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_t1', default='xgboost', help="choose models options:svm, xgboost, LogisticRegression")
    parser.add_argument('--method_t2', default='xgboost', help="choose models options:svm, xgboost, LogisticRegression")
    parser.add_argument('--method_t3', default='xgboost', help="choose models options:svm, xgboost, ensemble")
    # parser.add_argument('--Is_oversampling', type=bool, default='False', help="If over sampling")
    # parser.add_argument('--Is_downsampling', type=bool, default='True', help="If down sampling")
    opt = parser.parse_args()

    # configs:
    X_train_dir = 'data/train_features.csv'
    y_train_dir = 'data/train_labels.csv'
    X_test_dir = 'data/test_features.csv'
    y_pred_save_dir = './test_labels.csv'
    do_over_sampling = False
    data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    data_x_filled = fill_missing_data(data_x=data_x)
    data_x_test_filled = fill_missing_data(data_x=data_x_test)

    # test_ID = data_x_test_filled['pid']
    test_ID = []
    # print(len(data_x_test.index))
    for i in range(0, len(data_x_test.index)):
        if i % 12 == 0:
            test_ID.append(data_x_test.values[i, 0])
    test_ID = np.array(test_ID)
    print('The shape of test_ID is {}'.format(test_ID.shape))
    y_train = from_csv_to_ndarray(data=data_y)
    x_train = from_csv_to_ndarray(data=data_x_filled)
    x_test = from_csv_to_ndarray(data=data_x_test_filled)

    x_train = x_train.reshape(int(x_train.shape[0] / 12), int(x_train.shape[1] * 12))
    x_test = x_test.reshape(int(x_test.shape[0] / 12), int(x_test.shape[1] * 12))
    print("after reshape, x_train: {}".format(x_train.shape))
    print("after reshape, x_test: {}".format(x_test.shape))

    x_train, x_test = data_preprocessing(x_train, x_test)  # Normalizaiton and ...
    # if do_over_sampling:
    #     x_train, y_train = over_sampling(x_train, y_train)

    sub = pd.DataFrame()
    # for i in range(0, 10):
    for i in range(0, data_y.columns.shape[0] - 1):
        label = data_y.columns[i + 1]
        x_train_selected, y_train_selected, x_test_selected = select_feature(x_train, y_train[:, i], x_test)
        print('***************Predicting {}***************'.format(label))
        print()
        if 0 <= i < 10:  # subtask 1
            if opt.method_t1 == 'LogisticRegression':
                print('Using Logistic Regression...')
                clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
            elif opt.method_t1 == 'xgboost':
                print('Using XGBoost Classifier...')
                clf = xgb.XGBoostClassifier
            elif opt.method_t1 == 'svm':
                print('Using SVC...')
                clf = SVC(gamma='auto', class_weight='balanced')
            else:
                raise Exception('Please indicate learning method!')
            model = evaluate_model(clf, x_train_selected, y_train_selected, 'clf')
            prediction = model.predict(x_test_selected)
            print(prediction.shape)
            sub['pid'] = test_ID
            sub['{}'.format(label)] = prediction
        elif i == 10:  # subtask 2
            if opt.method_t2 == 'LogisticRegression':
                print('Using Logistic Regression...')
                clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
            elif opt.method_t2 == 'xgboost':
                print('Using XGBoost Classifier...')
                clf = xgb.XGBoostClassifier
            elif opt.method_t2 == 'svm':
                print('Using SVC...')
                clf = SVC(gamma='auto', class_weight='balanced')
            else:
                raise Exception('Please indicate learning method!')
            model = evaluate_model(clf, x_train_selected, y_train_selected, 'clf')
            prediction = model.predict(x_test_selected)
            print(prediction.shape)
            sub['pid'] = test_ID
            sub['{}'.format(label)] = prediction
        else:  # subtask 3
            if opt.method_t3 == 'svm':
                print('Using SVR...')
                clf = SVR(kernel='rbf', gamma='auto')
                model = evaluate_model(clf, x_train_selected, y_train_selected, 'reg')
                prediction = model.predict(x_test_selected)
                print(prediction.shape)
                sub['pid'] = test_ID
                sub['{}'.format(label)] = prediction
            elif opt.method_t3 == 'xgboost':
                print('Using xgboost...')
                clf = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                       learning_rate=0.05, max_depth=3,
                                       min_child_weight=1.7817, n_estimators=2200,
                                       reg_alpha=0.4640, reg_lambda=0.8571,
                                       subsample=0.5213, silent=1,
                                       random_state=7, nthread=-1)
                model = evaluate_model(clf, x_train_selected, y_train_selected, 'reg')
                prediction = model.predict(x_test_selected)
                print(prediction.shape)
                sub['pid'] = test_ID
                sub['{}'.format(label)] = prediction
            elif opt.method_t3 == 'ensemble':
                # =================================================
                # Ensemble + stacking
                # =================================================
                print()
                print("Ensemble start...")
                score = get_model_score(lasso, x_train_selected, y_train_selected, 5, 'esm')
                print("\nLasso score: {:.4f}\n".format(score))
                score = get_model_score(ENet, x_train_selected, y_train_selected, 5, 'esm')
                print("ElasticNet score: {:.4f}\n".format(score))
                score = get_model_score(KRR, x_train_selected, y_train_selected, 5, 'esm')
                print("Kernel Ridge score: {:.4f}\n".format(score))
                score = get_model_score(GBoost, x_train_selected, y_train_selected, 5, 'esm')
                print("Gradient Boosting score: {:.4f}\n".format(score))
                score = get_model_score(model_xgb, x_train_selected, y_train_selected, 5, 'esm')
                print("Xgboost score: {:.4f}\n".format(score))
                score = get_model_score(model_lgb, x_train_selected, y_train_selected, 5, 'esm')
                print("LGBM score: {:.4f}\n".format(score))

                stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                                 meta_model=lasso)
                score = get_model_score(stacked_averaged_models, x_train_selected, y_train_selected, 5, 'esm')
                print("Stacking Averaged models score: {:.4f}".format(score))
                stacked_averaged_models.fit(x_train_selected, y_train_selected)
                stacked_train_pred = stacked_averaged_models.predict(x_train_selected)
                stacked_pred = stacked_averaged_models.predict(x_test_selected)
                print(r2_score(y_train_selected, stacked_train_pred))
                model_xgb.fit(x_train_selected, y_train_selected)
                xgb_train_pred = model_xgb.predict(x_train_selected)
                xgb_pred = model_xgb.predict(x_test_selected)
                print(r2_score(y_train_selected, xgb_train_pred))
                model_lgb.fit(x_train_selected, y_train_selected)
                lgb_train_pred = model_lgb.predict(x_train_selected)
                lgb_pred = model_lgb.predict(x_test_selected)
                print(r2_score(y_train_selected, lgb_train_pred))
                print('RMSLE score on train data:')
                print(r2_score(y_train_selected, stacked_train_pred * 0.70 +
                               xgb_train_pred * 0.15 + lgb_train_pred * 0.15))
                prediction = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
                print(prediction.shape)
                sub['pid'] = test_ID
                sub['{}'.format(label)] = prediction

    sub.to_csv(y_pred_save_dir, index=False)
    score_submission(y_pred_save_dir)
    # suppose df is a pandas dataframe containing the result
    sub.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')


if __name__ == '__main__':
    main()
