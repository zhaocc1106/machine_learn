#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Use scikit-learn svm to classify.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/6/6 下午6:54
"""

# common libs.
import os

# 3rd-part libs.
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


def load_data_file(file_path):
    """Load data from testSet.txt.

    Args:
        file_path: the data file path.

    Returns: The data mat and label mat.
    """
    feature_arr = []
    label_arr = []
    fr = open(file_path)
    for line in fr.readlines():
        line_arr = line.strip().split()
        feature_arr.append([float(line_arr[0]), float(line_arr[1])])
        label_arr.append(float(line_arr[2]))
    return np.asarray(feature_arr), np.asarray(label_arr)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in.

    Args:
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

    Returns:
        xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Args:
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def show_best_fit_linear_clf(feature_arr, label_arr, weights, title):
    """Show the best fit linear classifier.

    Args:
        feature_arr: The feature array.
        label_arr: The label array.
        weights: The weights of x, y and bias. [bias_w, x_w, y_w]
        title: The image title.

    Returns:

    """
    n = np.shape(feature_arr)[0]

    # Generate the coordinates of 2 classes data.
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_arr[i]) == 1:
            xcord1.append(feature_arr[i, 0])
            ycord1.append(feature_arr[i, 1])
        else:
            xcord2.append(feature_arr[i, 0])
            ycord2.append(feature_arr[i, 1])

    # Plot all data points of 2 classes.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label="正样本")
    ax.scatter(xcord2, ycord2, s=30, c='green', label="负样本")

    # Plot classifier line.
    x = np.arange(-5.0, 10.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y.transpose(), label="最佳分割线")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.show()


def show_support_vectors(feature_arr, label_arr, support_vec, title):
    """Show training data and support vectors of training.

    Args:
        feature_arr: The train data feature array.
        label_arr: The train data label array.
        support_vec: The support vectors of training.
        title: The image title.

    Returns:

    """
    n = np.shape(feature_arr)[0]

    # Generate the coordinates of 2 classes data.
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_arr[i]) == 1:
            xcord1.append(feature_arr[i, 0])
            ycord1.append(feature_arr[i, 1])
        else:
            xcord2.append(feature_arr[i, 0])
            ycord2.append(feature_arr[i, 1])

    # Plot all data points of 2 classes.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label="正样本")
    ax.scatter(xcord2, ycord2, s=30, c='green', label="负样本")

    # Show support vectors.
    ax.scatter(support_vec[:, 0],
               support_vec[:, 1],
               s=150,
               c='none',
               alpha=0.7,
               linewidth=1.5,
               edgecolor='red', label="支持向量")

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.show()


def show_decision_boundaries(test_feature_arr, test_label_arr, clf, title):
    """Show the decision boundaries of classifier.

    Args:
        test_feature_arr: The test feature array.
        test_label_arr: The test label array.
        clf: The classifier.
        title: The image title.
    """
    n = np.shape(test_feature_arr)[0]

    # Generate the coordinates of 2 classes data.
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(test_label_arr[i]) == 1:
            xcord1.append(test_feature_arr[i, 0])
            ycord1.append(test_feature_arr[i, 1])
        else:
            xcord2.append(test_feature_arr[i, 0])
            ycord2.append(test_feature_arr[i, 1])

    # Plot all data points of 2 classes.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label="正样本")
    ax.scatter(xcord2, ycord2, s=30, c='green', label="负样本")

    # Get mesh grid of test features.
    xx, yy = make_meshgrid(test_feature_arr[:, 0], test_feature_arr[:, 1])

    # Plot contours.
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.title(title)
    plt.show()


def linear_svm_fit(feature_arr, label_arr):
    """Fit svm classifier.

    Args:
        feature_arr: The feature array.
        label_arr: The label array.

    Returns:
        The weights of x, y and bias. [bias_w, x_w, y_w]
    """
    svc = svm.LinearSVC()
    svc.fit(feature_arr, label_arr)
    weights = svc.intercept_.tolist()
    weights.extend(svc.coef_[0])
    return weights


def nonlinear_svm_fit_and_test(train_feature_arr, train_label_arr,
                               test_feature_arr, test_label_arr):
    """Fit nonlinear svm classifier and test.

    Args:
        train_feature_arr: The train feature array.
        train_label_arr: The train label array.
        test_feature_arr: The test feature array.
        test_label_arr: The test label array.

    Returns:
        The classifier.
    """
    # fit.
    clf = svm.SVC(gamma="scale")
    clf.fit(train_feature_arr, train_label_arr)

    # test.
    pre_label = clf.predict(test_feature_arr)
    err = pre_label != test_label_arr
    err_rate = np.sum(err) / np.shape(err)[0]
    print("The test error rate: ", str(err_rate))

    return clf


if __name__ == "__main__":
    # Linear svm.
    feature_arr, label_arr = load_data_file("testSet.txt")
    weights = linear_svm_fit(feature_arr, label_arr)
    show_best_fit_linear_clf(feature_arr, label_arr, weights, "线性svm")

    # nonlinear svm.
    train_feature_arr, train_label_arr = load_data_file("testSetRBF.txt")
    test_feature_arr, test_label_arr = load_data_file("testSetRBF2.txt")
    clf = nonlinear_svm_fit_and_test(train_feature_arr,
                                     train_label_arr,
                                     test_feature_arr,
                                     test_label_arr)
    # show training samples and support vectors.
    show_support_vectors(train_feature_arr, train_label_arr,
                         clf.support_vectors_, "训练样本与支持向量")
    # show test samples and decision boundaries of classifier.
    show_decision_boundaries(test_feature_arr, test_label_arr, clf, "测试样本与决策边界")
