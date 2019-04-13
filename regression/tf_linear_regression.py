#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Use tf.estimator.LinearRegressor to learn the linear function of ex data.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-4-8 下午8:24
"""

# Common libs.
import os
import shutil

# 3rd-part libs.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

_NUM_EXAMPLES = {
    'train': 200,
    'test': 200
}

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0]]

_CSV_COLUMNS = ['x', 'y']

_MODEL_DIR = '/tmp/regression/ex/'


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
        features_dict = dict(zip(_CSV_COLUMNS, columns[1:]))
        # Pop the y feature.
        y = features_dict.pop("y")
        return features_dict, y

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(map_func=parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    dataset = dataset.batch(batch_size)
    return dataset


def train_and_evaluate(train_data, test_data):
    """Train and evaluate. Calc the bias and x weight of linear function(y =
    x * w + b).

    Args:
        train_data: The train data file.
        test_data: The test data file.

    Returns:
        The bias and x weight.
    """
    if os.path.exists(_MODEL_DIR):
        shutil.rmtree(_MODEL_DIR)

    # c_feature = tf.feature_column.numeric_column('c')
    x_feature = tf.feature_column.numeric_column('x')
    features = [x_feature]
    regressor = tf.estimator.LinearRegressor(
        feature_columns=features,
        model_dir=_MODEL_DIR,
        config=tf.estimator.RunConfig(save_summary_steps=100,
                                      save_checkpoints_steps=100),
        optimizer='Ftrl' # Use Follow-the-regularized-Leader algorithm.
    )

    def train_input_fn():
        return input_function(train_data, 10, True, 200)

    def test_input_fn():
        return input_function(test_data, 10, False, 1)

    # Train the regressor.
    regressor.train(input_fn=train_input_fn)

    # Evaluate the regressor.
    scores = regressor.evaluate(input_fn=test_input_fn)
    print("scores: \n", str(scores))

    train_vars = regressor.get_variable_names()
    print("train_vars:\n", str(train_vars))
    bias = regressor.get_variable_value("linear/linear_model/bias_weights")
    x_weight = regressor.get_variable_value("linear/linear_model/x/weights")
    print("bias: {0}, x_weight: {1}".format(bias, x_weight))
    return bias, x_weight


def show_regres(data_arr, labels, w):
    """
    Show 2D data and its best fit straight line.

    Args:
        data_arr: The features data array.
        labels: The labels.
        w: The weights(the column vector of x weight and bias).

    """
    fit = plt.figure()
    ax = fit.add_subplot(111)

    # Show all points.
    ax.scatter((np.array(data_arr).T)[1], np.array(labels), s=10, c="red")

    # Show the best fit straight line.
    # Y = X * w
    X = np.arange(0.0, 1.0, 0.01)
    X_mat = np.mat(X)
    data_mat = np.mat(np.zeros((100, 2)))
    data_mat[:, 0] = 1.0
    data_mat[:, 1] = X_mat.T
    Y = np.transpose(data_mat * np.mat(w)).tolist()[0]
    ax.plot(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def load_data(file):
    """Read features data and labels from file.

    Args:
        file: The file path.

    Returns:
        The features data array and labels.
    """
    print("load_data ", file)
    data_arr = []
    labels = []
    fr = open(file)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        length = len(line_arr)
        tmp_arr = []
        for i in range(length - 1):
            tmp_arr.append(float(line_arr[i]))
        data_arr.append(tmp_arr)
        labels.append(float(line_arr[-1]))
    return data_arr, labels


if __name__ == "__main__":
    # Transform data file to csv file if not transformed.
    if not os.path.exists('./ex0.csv'):
        trans_to_csv('./ex0.txt', './ex0.csv')
    if not os.path.exists('./ex1.csv'):
        trans_to_csv('./ex1.txt', './ex1.csv')

    """
    # Test input function.
    dataset = input_function('./ex0.csv', 10, True, 10)
    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Run initializer.
    sess = tf.InteractiveSession()
    sess.run(iterator.initializer)

    # Iter.
    while True:
        try:
            print("next element:")
            feature_columns, y = sess.run(next_element)
            print("feature_columns:\n", str(feature_columns))
            print("y:\n", str(y))
        except tf.errors.OutOfRangeError:
            print("out of range.")
            break
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    bias, x_weight = train_and_evaluate('./ex0.csv', 'ex1.csv')
    ws = np.asarray([bias, x_weight[0]])
    data_arr, labels = load_data('./ex1.txt')
    show_regres(data_arr, labels, ws)
