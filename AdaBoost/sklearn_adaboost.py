#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Use scikit-learn adaboost to train and test.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2020/1/14 上午9:01
"""

# 3rd-part libs.
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def score(y_true, y_pred):
    """Score the model.

    Args:
        y_true: The true y data.
        y_pred: The prediction y data.
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recal = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print("conf_mat: \n", str(conf_mat))
    print("prec: \n", str(prec))
    print("recal: \n", str(recal))
    print("roc_auc: \n", str(roc_auc))


def train_and_eval(train_x, train_y, eval_x, eval_y):
    """Train and evaluation.

    Args:
        train_x: Training x.
        train_y: Training y.
        eval_x: Evaluation x.
        eval_y: Evaluation y.
    """
    ada_clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=0.5)
    ada_clf.fit(train_x, train_y)
    print("**********Train score**********")
    y_delta = ada_clf.predict(train_x)
    score(train_y, y_delta)
    print("**********Test score**********")
    y_delta = ada_clf.predict(eval_x)
    score(eval_y, y_delta)

    # Grid search for param n_estimators.
    param_grid = [{"n_estimators": [100, 200, 300]}]
    grid_search = GridSearchCV(ada_clf, param_grid, cv=5, scoring="roc_auc",
                               verbose=1)
    grid_search.fit(train_x, train_y)
    ada_clf = grid_search.best_estimator_
    print("**********Train score**********")
    y_delta = ada_clf.predict(train_x)
    score(train_y, y_delta)
    print("**********Test score**********")
    y_delta = ada_clf.predict(eval_x)
    score(eval_y, y_delta)


if __name__ == "__main__":
    train_data = np.loadtxt("./horseColicTraining2.txt")
    eval_data = np.loadtxt("./horseColicTest2.txt")
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    eval_x = eval_data[:, :-1]
    eval_y = eval_data[:, -1]
    train_and_eval(train_x, train_y, eval_x, eval_y)
