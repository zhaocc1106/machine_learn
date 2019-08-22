#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
One simple softmax regression network algorithm.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/10/22 20:39
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

    def __init__(self, mini_batch):
        """The construct function of network.

        :param mini_batch: The mini batch.
        """
        self.mini_batch = mini_batch

        # Initialize the params use zeros.
        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # Define the input and desired output. The input is a n * 784 vector.
        # The output is a n * 10 vector.
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        # Define the really output.
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

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
                train_step.run({self.x: batch_xs, self.y_: batch_ys})

            # Calc the training accuracy.
            training_accuracy = accuracy.eval({self.x:
                                                   training_data.images,
                                               self.y_:
                                                   training_data.labels})
            # Calc the validation accuracy.
            validation_accuracy = accuracy.eval({self.x:
                                                     validation_data.images,
                                                 self.y_:
                                                     validation_data.labels})
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
                                               self.y_: test_data.labels})
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
    mnist = input_data.read_data_sets("datas/MNIST_data/", one_hot=True)
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
    network = Network(mini_batch=10)
    training_accuracys, validation_accuracys, test_accuracy = network.SGD(
        training_data=training_data,
        eta=0.1,
        epoches=20,
        validation_data=validation_data,
        test_data=test_data)
    plot_accuracy(training_accuracys, validation_accuracys, test_accuracy)
