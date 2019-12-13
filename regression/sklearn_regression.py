#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Build regression model by scikit-learn.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/12/11 上午9:23
"""

# common libs.
import os

# 3rd-part libs.
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing


def trans_to_csv(data_file, output_file):
    """Transform data file to csv file.

    Args:
        data_file: The data file.
        output_file: The output file.
    """
    assert os.path.exists(data_file), ('%s not found' % data_file)

    with open(output_file, 'w') as output, open(data_file, 'r') as input:
        # for line in input.readlines():
        # print('line: ', line.replace('\t', ','))
        # output.write(line.replace('\t', ','))
        lines = input.readlines()
        output.writelines(map(lambda line: line.replace('\t', ','), lines))
    print("Transform finished.")


def train_and_evaluate(train_x, train_y, test_x, test_y):
    """Train and evaluate."""
    linear_reg = linear_model.SGDRegressor(verbose=1, max_iter=5, penalty="l2")
    linear_reg.fit(train_x, train_y)

    train_score = linear_reg.score(train_x, train_y)
    test_score = linear_reg.score(test_x, test_y)
    print("coef: {0}, intercept: {1}".format(str(linear_reg.coef_),
                                             str(linear_reg.intercept_)))

    print("train score: {0}, test score: {1}".format(train_score, test_score))


if __name__ == "__main__":
    # Convert to csv if not exists.
    if not os.path.exists("ex0.csv"):
        trans_to_csv("ex0.txt", "ex0.csv")
    if not os.path.exists("ex1.csv"):
        trans_to_csv("ex1.txt", "ex1.csv")

    # Load x and y.
    train_data = np.asarray(pd.read_csv("ex0.csv"))
    test_data = np.asarray(pd.read_csv("ex1.csv"))
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]

    # train_x = preprocessing.StandardScaler().fit_transform(train_x)
    # print(train_x)
    # test_x = preprocessing.StandardScaler().fit_transform(test_x)
    # print(test_x)

    # Train and evaluate.
    train_and_evaluate(train_x, train_y, test_x, test_y)
