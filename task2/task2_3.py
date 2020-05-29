import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import argparse
import warnings
import math
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from score_submission import score_submission
from sklearn.model_selection import KFold
from sklearn.base import clone
from scipy.special import expit


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn


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
    print(data_x.median())
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
    clf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=0, class_weight="balanced")
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
    # return x_train, y_train, x_test


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
    parser.add_argument('--method_t1', default='svm', help="choose models options:svm, xgboost, LogisticRegression")
    parser.add_argument('--method_t2', default='svm', help="choose models options:svm, xgboost, LogisticRegression")
    parser.add_argument('--method_t3', default='xgboost', help="choose models options:svm, xgboost, ensemble")
    parser.add_argument('--select_feature_flag', default=False, help="select feature or not")
    # parser.add_argument('--Is_oversampling', type=bool, default='False', help="If over sampling")
    # parser.add_argument('--Is_downsampling', type=bool, default='True', help="If down sampling")
    opt = parser.parse_args()

    # configs:
    X_train_dir = 'data/train_features.csv'
    y_train_dir = 'data/train_labels.csv'
    X_test_dir = 'data/test_features.csv'
    y_pred_save_dir = './test_labels.csv'
    data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    # data_x_filled = fill_missing_data(data_x=data_x)
    # data_x_test_filled = fill_missing_data(data_x=data_x_test)

    # test_ID = data_x_test['pid']

    test_ID = []
    # print(len(data_x_test.index))
    for i in range(0, len(data_x_test.index)):
        if i % 12 == 0:
            test_ID.append(data_x_test.values[i, 0])
    test_ID = np.array(test_ID)
    print('The shape of test_ID is {}'.format(test_ID.shape))
    y_train = from_csv_to_ndarray(data=data_y)
    x_train = from_csv_to_ndarray(data=data_x)
    x_test = from_csv_to_ndarray(data=data_x_test)

    x_train = x_train.reshape(int(x_train.shape[0] / 12), int(x_train.shape[1]*12))
    x_test = x_test.reshape(int(x_test.shape[0] / 12), int(x_test.shape[1]*12))
    print("after reshape, x_train: {}".format(x_train.shape))
    print("after reshape, x_test: {}".format(x_test.shape))

    # filling missing value
    col_mean_train = np.nanmean(x_train, axis=0)
    col_mean_test = np.nanmean(x_test, axis=0)
    # Find indices that need to replace
    inds_train = np.where(np.isnan(x_train))
    inds_test = np.where(np.isnan(x_test))
    # Place column means in the indices. Align the arrays using take
    x_train[inds_train] = np.take(col_mean_train, inds_train[1])
    x_test[inds_test] = np.take(col_mean_test, inds_test[1])

    x_train, x_test = data_preprocessing(x_train, x_test)  # Normalizaiton and ...

    sub = pd.DataFrame()
    for i in range(0, data_y.columns.shape[0] - 1):
        label = data_y.columns[i + 1]
        if opt.select_feature_flag:
            x_train_selected, y_train_selected, x_test_selected = select_feature(x_train, y_train[:, i], x_test)
        else:
            x_train_selected=x_train
            y_train_selected= y_train[:, i]
            x_test_selected = x_test
        print('***************Predicting {}***************'.format(label))
        print()
        if 0 <= i < 10:  # subtask 1
            if opt.method_t1 == 'LogisticRegression':
                print('Using Logistic Regression...')
                clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
            elif opt.method_t1 == 'svm':
                print('Using SVC...')
                clf = SVC(gamma='auto', class_weight='balanced')
            else:
                raise Exception('Please indicate learning method!')
            model = evaluate_model(clf, x_train_selected, y_train_selected, 'clf')
            # prediction = model.predict(x_test_selected)
            prediction_score = model.decision_function(x_test_selected)
            prediction = []
            for i in range(0, len(prediction_score)):
                prediction.append(expit(prediction_score[i]))
            prediction = np.array(prediction)
            # prediction = predictor.predict(x_test_selected)
            print(prediction.shape)
            sub['pid'] = test_ID
            sub['{}'.format(label)] = prediction
        elif i == 10:  # subtask 2
            if opt.method_t2 == 'LogisticRegression':
                print('Using Logistic Regression...')
                clf = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
            elif opt.method_t2 == 'svm':
                print('Using SVC...')
                clf = SVC(gamma='auto', class_weight='balanced')
            else:
                raise Exception('Please indicate learning method!')
            model = evaluate_model(clf, x_train_selected, y_train_selected, 'clf')
            # prediction = model.predict(x_test_selected)
            prediction_score = model.decision_function(x_test_selected)
            prediction = []
            for i in range(0, len(prediction_score)):
                prediction.append(expit(prediction_score[i]))
            prediction = np.array(prediction)
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

    sub.to_csv(y_pred_save_dir, index=False)
    score_submission(y_pred_save_dir)
    # suppose df is a pandas dataframe containing the result
    sub.to_csv('prediction_1.zip', index=False, float_format='%.3f', compression='zip')


if __name__ == '__main__':
    main()
