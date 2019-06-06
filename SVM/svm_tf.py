#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Test tensorflow svm(Only support linear svm).

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/6/5 22:51
"""

# common libs
import os
import shutil

# 3rd-part libs
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import matplotlib.pyplot as plt

_COLUMNS = ['x', 'y', 'label']
_MODEL_DIR_TEST_SET = '/tmp/svm/testSet/'
_MODEL_DIR_TEST_SET_RBF = '/tmp/svm/testSetRBF/'


def input_func(data_file):
    """The input function.

    Args:
        data_file: The data file path.

    Returns:
        tensor columns.
    """
    print("parsing: ", data_file)
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    fr = open(data_file)

    # Read line to arr.
    data = []
    label = []
    example_ids = []
    for i, line in enumerate(fr.readlines()):
        line_arr = line.strip().split("\t")
        data.append([float(line_arr[0]), float(line_arr[1])])
        example_ids.append(str(i))
        # convert -1 to 0
        label.append([float(line_arr[2])]
                     if float(line_arr[2]) == 1 else [float(0.0)])

    data_arr = np.asarray(data)
    label_arr = np.asarray(label)

    # Convert to tensor.
    feature_columns = {
        _COLUMNS[i]:
            tf.reshape(
                tf.convert_to_tensor(data_arr[:, i]), [-1, 1])
        for i in range(len(_COLUMNS) - 1)
    }
    feature_columns['example_ids'] = tf.convert_to_tensor(example_ids)
    labels_column = tf.convert_to_tensor(label_arr)

    print(feature_columns)
    print(labels_column)
    return feature_columns, labels_column


def train_and_eval(train_data, test_data, model_dir):
    """Training and evaluation.

    Args:
        train_data: The training data file path.
        test_data: The test data file path.
        model_dir: The model saver directory.

    Returns:

    """
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Define feature columns
    x_feature = tf_contrib.layers.real_valued_column('x')
    y_feature = tf_contrib.layers.real_valued_column('y')
    features = [x_feature, y_feature]

    # Build SVM.
    # tensorflow svm don't support non-linear svm currently.
    svm = tf_contrib.learn.SVM(
        example_id_column='example_ids',
        feature_columns=features,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(save_summary_steps=100,
                                      save_checkpoints_steps=100),
    )

    # Define train input fn.
    def train_input_fn():
        return input_func(train_data)

    # Define test input fn.
    def test_input_fn():
        return input_func(test_data)

    # Fit svm.
    svm.fit(input_fn=train_input_fn, steps=30)

    # Test svm.
    scores = svm.evaluate(input_fn=test_input_fn, steps=1)
    print("scores:\n", str(scores))

    train_vars = svm.get_variable_names()
    print("train_vars:\n", str(train_vars))

    # Get weights and bias.
    x_weight = svm.get_variable_value('linear/x/weight')
    y_weight = svm.get_variable_value('linear/y/weight')
    bias = svm.get_variable_value('linear/bias_weight')

    return [bias, x_weight, y_weight]


def show_best_fit(weights):
    """Show the best fit linear classifier of testSet.

    Args:
        weights: The weights of x, y and bias. [bias_w, x_w, y_w]

    Returns:

    """
    data_mat, label_mat = load_data_file()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]

    # Generate the coordinates of 2 classes data.
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_mat[i][1])
            ycord1.append(data_mat[i][2])
        else:
            xcord2.append(data_mat[i][1])
            ycord2.append(data_mat[i][2])

    # Plot all data points of 2 classes.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot classifier line.
    x = np.arange(-5.0, 10.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def load_data_file():
    """Load data from testSet.txt.

    Returns: The data mat and label mat.
    """
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data_mat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label_mat.append(int(lineArr[2]))
    return data_mat, label_mat


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # input_func("./testSet.txt")
    weights_bias = train_and_eval("./testSet.txt", "./testSet.txt",
                                  _MODEL_DIR_TEST_SET)
    show_best_fit(weights_bias)