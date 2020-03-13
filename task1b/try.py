import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest


def isnan(x):
    return x != x


def read_data(path='./data/train.csv'):
    data = pd.read_csv(path).drop('Id', axis=1)
    data_x = np.array(data, float)[:, 1:]
    data_y = np.array(data, float)[:, 0]
    return data_x, data_y


def feature_trans(x):
    feature = np.hstack([x, x * x, np.exp(x), np.cos(x), np.ones([x.shape[0], 1])])
    print("feature shape: {}".format(feature.shape))
    return feature


def feature_selection_ps(data_x, data_y):
    corr = np.zeros(data_x.shape[1])
    corr2 = np.zeros(data_x.shape[1])
    # label= np.squeeze(y)
    for i in range(data_x.shape[1]):
        corr[i], _ = pearsonr(data_x[:, i], data_y[:])
        corr2[i], _ = spearmanr(data_x[:, i], data_y[:])
        if isnan(corr[i]):
            corr[i] = 0
        corr[i] = abs(corr[i])
        if isnan(corr2[i]):
            corr2[i] = 0
        corr2[i] = abs(corr2[i])

    # set up mask for vars with corrilation over 0.1
    mask1 = np.int64(corr > 0.05)  # Pearson mask: lin correlation with y over 0.1
    print(corr)
    mask2 = np.int64(corr2 > 0.05)  # Searman mask: any correlation with y over 0.1
    mask = mask1 | mask2
    # index= np.nonzero(mask)
    print(mask)
    num_feature = np.sum(mask)  # number of selected features
    print('Final feature numbers is %d!' % num_feature)

    # Use index to trim data down to relevant features
    data_x_selected = data_x * mask
    return data_x_selected, data_y


def feature_selection_kbest(x_train, y_train, feature_num):
    print()
    print("Selecting features...")
    print("Before feature selection: {}".format(x_train.shape))
    # x_selected = np.zeros((x_train.shape[0],21)) # nx21
    x_new = SelectKBest(score_func=f_regression, k=feature_num).fit_transform(x_train, y_train)
    print("After feature selection: {}".format(x_new.shape))
    for i in range(x_train.shape[1]):
        not_finded = True
        for j in range(x_new.shape[1]):
            if (x_train[:, i] == x_new[:, j]).all():
                not_finded = False
        if not_finded:
            x_train[:, i] = 0
    print(x_train)
    return x_train, y_train


def try_different_method(model, x_train, y_train, x_test, y_test, score_func):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    return score_func(y_test, result)


def find_best_param_show_error(data_x, data_y, score_func, n_split, alphas):
    # CV: select best alpha
    # alphas=[1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1 ]
    count = np.zeros(len(alphas))
    error = np.zeros(len(alphas))
    # for j in range(200):

    kf = KFold(n_splits=n_split, shuffle=True)
    for train_index, val_index in kf.split(data_x):
        data_x_tr, data_x_val = data_x[train_index], data_x[val_index]
        data_y_tr, data_y_val = data_y[train_index], data_y[val_index]
        for i in range(len(alphas)):
            clf = Ridge(alphas[i])
            clf.fit(data_x_tr, data_y_tr)
            data_y_pred = clf.predict(data_y_val)
            error[i] += np.sqrt(mean_squared_error(data_y_val, data_y_pred)) / n_split / 200
    count[np.argmin(error)] += 1
    for i in range(len(alphas)):
        print("error is %f when alpha=%f" % (error[i], alphas[i]))
        print("achieve minimum %d times when alpha=%f" % (count[i], alphas[i]))


def find_best_param(data_x, data_y, score_func, n_split, alphas):
    score_mean = 0
    best_score = 0
    best_model_param = 0
    kf = KFold(n_splits=n_split, shuffle=True)
    for alpha_ridge in [0.0001, 0.0005, 0.001, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.01, 0.05, 0.1, 0.5, 1, 2,
                        10, 20, 30, 40, 50]:
        for train_idx, test_idx in kf.split(data_x):
            x_train = data_x[train_idx]
            y_train = data_y[train_idx]
            x_test = data_x[test_idx]
            y_test = data_y[test_idx]
            score_mean += try_different_method(Ridge(alpha=alpha_ridge), x_train, y_train, x_test, y_test,
                                               mean_squared_error)
        score_mean /= n_split
        if best_score < score_mean:
            best_score = score_mean
            best_model_param = alpha_ridge
    print("best model parameter for ridge is {}".format(best_model_param))
    return best_model_param


def main():
    data_x, data_y = read_data(path='./data/train.csv')
    x_train = feature_trans(data_x)
    y_train = data_y
    # use feature_selection_ps or feature_selection_kbest
    x_selected, y_selected = feature_selection_ps(x_train, y_train)
    # x_selected, y_selected = feature_selection_kbest(x_train, y_train, 8)
    score_function = mean_squared_error
    find_best_parameter = False
    if find_best_parameter:
        best_param = find_best_param(data_x=x_selected, data_y=y_selected, score_func=score_function,
                                     n_split=5, alphas=[0.1, 0.5, 1, 5, 10, 15, 20, 25, 30, 40])
        clf = Ridge(best_param)
        clf.fit(x_selected, y_selected)
        print(clf.coef_)
        final_weight = clf.coef_
    else:
        clf = Ridge(10)  # tunning manually
        clf.fit(x_selected, y_selected)
        print("The final weight:")
        print(clf.coef_)
        final_weight = clf.coef_

    # np.savetxt('weight.csv', final_weight)
    data = pd.DataFrame(np.transpose(final_weight))
    # change the output file name here, also include some parameter information in the description please
    data.to_csv('./output.csv', index=False, header=False)


if __name__ == '__main__':
    main()
