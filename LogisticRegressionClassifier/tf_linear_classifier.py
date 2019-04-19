#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Use tf.estimator.LinearClassifier to learn the classifier of testSet.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-4-17 下午8:45
"""

# common libs.
import os
import shutil

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

_NUM_EXAMPLES = {
    'train': 100,
    'test': 100
}

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0]]

_CSV_COLUMNS = ['x', 'y', 'label']

_MODEL_DIR = '/tmp/logistic_regression_classifier/testSet/'


def trans_to_csv(data_file, output_file):
    """Transform data file to csv file.

    Args:
        data_file: The data file.
        output_file: The output file.
    """
    assert tf.gfile.Exists(data_file), ('%s not found' % data_file)

    with open(output_file, 'w') as output, open(data_file, 'r') as input:
        # for line in input.readlines():
        # print('line: ', line.replace('\t', ','))
        # output.write(line.replace('\t', ','))
        lines = input.readlines()
        output.writelines(map(lambda line: line.replace('\t', ','), lines))
    print("Transform finished.")


def input_function(data_file, batch_size, shuffle, num_epochs):
    """

    Args:
        data_file: The data file path.
        batch_size: The batch size.
        shuffle: If shuffle data.
        num_epochs: The number of epochs.

    Returns:
        The dataset of features and label columns.
    """
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    def parse_csv(line):
        print("Parsing ", data_file)
        # Decode one line.
        columns = tf.decode_csv(records=line,
                                record_defaults=_CSV_COLUMN_DEFAULTS)
        # Build features dict.
        features_dict = dict(zip(_CSV_COLUMNS, columns))
        # Pop the label feature.
        label = features_dict.pop("label")
        return features_dict, label

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(map_func=parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    dataset = dataset.batch(batch_size)
    return dataset


def train_and_evaluate(train_data, test_data):
    """Train and evaluate. Calc the bias and x weight of linear classifier
    function(y = x * w + b).

    Args:
        train_data: The train data file.
        test_data: The test data file.

    Returns:
        The weights of bias, x and y.
    """
    if os.path.exists(_MODEL_DIR):
        shutil.rmtree(_MODEL_DIR)

    x_feature = tf.feature_column.numeric_column('x')
    y_feature = tf.feature_column.numeric_column('y')
    features = [x_feature, y_feature]
    classifier = tf.estimator.LinearClassifier(
        feature_columns=features,
        model_dir=_MODEL_DIR,
        config=tf.estimator.RunConfig(save_summary_steps=100,
                                      save_checkpoints_steps=100),
        optimizer='Ftrl'  # Use Follow-the-regularized-Leader algorithm.
    )

    def train_input_fn():
        return input_function(train_data, 10, True, 200)

    def test_input_fn():
        return input_function(test_data, 10, False, 1)

    # Train the classifier.
    classifier.train(input_fn=train_input_fn)

    # Evaluate the classifier.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print("scores: \n", str(scores))

    train_vars = classifier.get_variable_names()
    print("train_vars:\n", str(train_vars))
    bias_w = classifier.get_variable_value("linear/linear_model/bias_weights")
    x_w = classifier.get_variable_value("linear/linear_model/x/weights")
    y_w = classifier.get_variable_value("linear/linear_model/y/weights")
    print("bias_weight:{0}, x_weight:{1}, y_weight:{2}".format(bias_w, x_w,
                                                               y_w))
    weights = [bias_w, x_w, y_w]
    return weights


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
    x = np.arange(-3.0, 3.0, 0.1)
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
    trans_to_csv("./testSet.txt", "./testSet.csv")
    weights = train_and_evaluate("./testSet.csv", "./testSet.csv")
    show_best_fit(weights)
