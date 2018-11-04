#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The simple multi layer perceptron algorithm.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/10/28 16:40
"""

# 3rd-part lib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


class Network(object):
    """
    Main class to construct and train network.
    """

    def __init__(self, sizes, drop_keep_p, mini_batch):
        """The construct function of network.

        :param sizes: The width of every layer.
        :drop_keep_p: The training dropout probability of hidden layer.
        :param mini_batch: The mini batch.
        """
        self.drop_keep_p = tf.placeholder(tf.float32)
        self.training_drop_keep_p = drop_keep_p
        self.mini_batch = mini_batch

        # Initialize all params.
        self.params = dict()
        self.params['w'] = []
        self.params['b'] = []
        # Firstly, initialize the hidden layer params.
        for i in range(1, len(sizes) - 1):
            self.params['w'].append(tf.Variable(
                tf.truncated_normal([sizes[i - 1], sizes[i]],
                                    stddev=(1.0 / tf.sqrt(tf.cast(sizes[i],
                                                                  tf.float32)))
                                    )))
            self.params['b'].append(tf.Variable(
                tf.zeros([sizes[i]], dtype=tf.float32)
            ))
        # Secondly, initialize the softmax layer params.
        self.params['w'].append(tf.Variable(tf.zeros([sizes[-2], sizes[-1]],
                                                     dtype=tf.float32)))
        self.params['b'].append(tf.Variable(tf.zeros([sizes[-1]],
                                                     dtype=tf.float32)))

        print("self.params:")
        print(self.params)

        # Define the input and desired output. The input is a n * 784 vector.
        # The output is a n * 10 vector.
        self.x = tf.placeholder(tf.float32, [None, sizes[0]])
        self.y_ = tf.placeholder(tf.float32, [None, sizes[-1]])

        # Define the really output.
        # Firstly, define the every hidden layer output.
        output_pre_layer = self.x
        for i in range(1, len(sizes) - 1):
            output_pre_layer = tf.nn.relu(
                tf.matmul(output_pre_layer,
                          self.params['w'][i - 1]) + self.params['b'][i - 1])
            # Dropout.
            output_pre_layer = tf.nn.dropout(output_pre_layer,
                                             self.drop_keep_p)
        # Secondly, define the softmax layer output.
        self.y = tf.nn.softmax(tf.matmul(output_pre_layer, self.params['w'][
            -1]) + self.params['b'][-1])

        # Define the cost use cross entropy.
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y),
                                                  reduction_indices=[1]))

    def SGD(self, training_data, eta, epoches, validation_data, test_data):
        """Training the network using stochastic gradient descent.

        :param training_data: The training data.
        :param eta: The learning rate.
        :param epoches: The epoch.
        :param test_data: The test data.
        :param validation_data: The validation data.
        :param test_data: The test data.
        """
        # Define the train step.
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.cost)

        # Define the accuracy.
        correct_prediction = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(
            self.y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize the tensor flow global params.
        tf.global_variables_initializer().run()

        # Calc the batches number of training, validation and test data.
        num_train_mini_batches = (int)(training_data.images.shape[0] / \
                                       self.mini_batch)
        num_validation_mini_batches = (int)(validation_data.images.shape[0]
                                            / \
                                            self.mini_batch)
        num_test_mini_batches = (int)(test_data.images.shape[0] / \
                                      self.mini_batch)

        print("num_train_mini_batches:")
        print(num_train_mini_batches)
        print("num_validation_mini_batches:")
        print(num_validation_mini_batches)
        print("num_test_mini_batches:")
        print(num_test_mini_batches)

        best_validation_accuracy = 0.0
        training_accuracys = []
        validation_accuracys = []
        for epoch in range(epoches):
            print("epoch {0} begin##########################".format(epoch))
            for i in range(num_train_mini_batches):
                batch_xs, batch_ys = training_data.next_batch(self.mini_batch)
                train_step.run({self.x: batch_xs,
                                self.y_: batch_ys,
                                self.drop_keep_p: self.training_drop_keep_p})

            # Calc the training accuracy.
            training_accuracy = accuracy.eval({self.x:
                                                   training_data.images,
                                               self.y_:
                                                   training_data.labels,
                                               self.drop_keep_p:
                                                   1.0})
            # Calc the validation accuracy.
            validation_accuracy = accuracy.eval({self.x:
                                                     validation_data.images,
                                                 self.y_:
                                                     validation_data.labels,
                                                 self.drop_keep_p:
                                                     1.0})
            training_accuracys.append(training_accuracy)
            validation_accuracys.append(validation_accuracy)
            print("Epoch {0}, training_accuracy: {1:.2%}, "
                  "validation_accuracy: {2:.2%}".format(epoch,
                                                        training_accuracy,
                                                        validation_accuracy))
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                # Calc the test accuracy.
                test_accuracy = accuracy.eval({self.x: test_data.images,
                                               self.y_: test_data.labels,
                                               self.drop_keep_p: 1.0})
                print("Get better validation_accuracy, test_accuracy:{"
                      "0:.2%}".format(test_accuracy))
        print("Training finished, the best validation accuracy: {0:.2%} with "
              "test accuracy: {1:.2%}".format(best_validation_accuracy,
                                              test_accuracy))
        return training_accuracys, validation_accuracys, test_accuracy


def plot_accuracy(training_accuracy, evaluation_accuracy, best_test_accuracy):
    """Plot the accuracy change of all epochs.

    Args:
        training_accuracy: The training accuracy list.
        evaluation_accuracy: The evaluation accuracy list.
        best_test_accuracy: The best test accuracy.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, len(training_accuracy), 1)
    ax.plot(x, training_accuracy, label='train accuracy')
    ax.plot(x, evaluation_accuracy, label='validation accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("The best test accuracy:{0:.2%}".format(best_test_accuracy))
    plt.legend()
    plt.show()


def load_mnist_data():
    """Load mnist data.

    :return: mnist data.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist.train, mnist.validation, mnist.test


if __name__ == "__main__":
    training_data, validation_data, test_data = load_mnist_data()
    print("training_data images shape:")
    print(training_data.images.shape)
    print("training_data labels shape:")
    print(training_data.labels.shape)
    print("validation_data images shape:")
    print(validation_data.images.shape)
    print("validation_data labels shape:")
    print(validation_data.labels.shape)
    print("test_data images shape:")
    print(test_data.images.shape)
    print("test_data labels shape:")
    print(test_data.labels.shape)

    sess = tf.InteractiveSession()  # register default session.
    network = Network(sizes=[784, 300, 10], drop_keep_p=0.5, mini_batch=10)
    training_accuracys, validation_accuracys, test_accuracy = network.SGD(
        training_data=training_data,
        eta=0.01,
        epoches=50,
        validation_data=validation_data,
        test_data=test_data)
    plot_accuracy(training_accuracys, validation_accuracys, test_accuracy)
