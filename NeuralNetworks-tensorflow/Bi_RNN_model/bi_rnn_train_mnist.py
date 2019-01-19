#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""The Bi-RNN(Bidirectional Recurrent Neural Networks) model to train MNIST
data.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/21 9:36
"""
# Common libs.
import time

# 3rd-part libs.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data():
    """Load mnist data.

    :return: mnist data.
    """
    print("load_mnist_data")
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


class Config(object):
    """The config.

    Attributes:
        learning_rate: The learning rate.
        max_samples: The max number of training samples.
        batch_size: The mini batch size.
        display_step: Show training results every display_steps.
        n_input: The number of input size.
        n_steps: The unrolled steps of LSTM.
        n_hidden: The hidden layer size.
        n_classes: The classes number of mnist.
    """
    learning_rate = 0.01
    batch_size = 128
    display_step = 10
    n_input = 28
    n_steps = 28
    n_hidden = 256
    n_classes = 10


class BiRNN(object):
    """The Bi-RNN

    Attributes:
        __config: The model config.
        __costs: The costs.
        __train_op: The train operation.
    """

    def __init__(self, config):
        """ Init the Bi-RNN by config. Inference the graph of Bi-RNN.

        Args:
            config: The model config.
        """
        self.__config = config

        self.__x = tf.placeholder(dtype=tf.float32, shape=[None, config.n_steps,
                                                           config.n_input],
                                  name="input")
        self.__y = tf.placeholder(dtype=tf.float32, shape=[None,
                                                           config.n_classes],
                                  name="class")
        self.__costs, self.__train_op, self.__accuracy = self.__inference(
            self.__x, self.__y)

    def __inference(self, x, y):
        """Construct the graph of Bi-RNN. Inference the costs and train
        operaion.

        Args:
            x: The mnist image input.
            y: The mnist image class.

        Returns:
            costs: The costs of the Bi-RNN model.
            train_op: The train operation of the Bi-RNN model.
            accuracy: The accuracy of the Bi-RNN model.
        """
        x = tf.transpose(x, [1, 0, 2])
        print(x)
        x = tf.reshape(x, [-1, self.__config.n_input])
        print(x)
        x = tf.split(x, self.__config.n_steps)
        print(x)
        """
        The initial input structure is as flowing. shape is
        batch_size * height * weight (3D):
        weight: 28
        height: 28
        batch_size: 128
        ↑        ←   w   →
        |      ↑ ×××××××××
        |        ×××××××××
        |      h ×××××××××
        |        ×××××××××
        |      ↓ ×××××××××
        |
        b        ←   w   →
        a      ↑ ×××××××××
        c        ×××××××××
        t      h ×××××××××
        h        ×××××××××
        |      ↓ ×××××××××
        |
        |        ←   w   →
        |      ↑ ×××××××××
        |        ×××××××××
        |      h ×××××××××
        |        ×××××××××
        |      ↓ ×××××××××
        ↓           ...


        The final input structure is as flowing, shape is
        n_step * (batch_size * weight) (2D):
        weight: 28
        n_step: 28 (rnn unrolled step)
        batch_size: 128
        ↑       ←-----------------batch----------------→
        |       ←   w   →←   w   →←   w   →←   w   → ...
        |       ****************************************
        |
        |       ←-----------------batch----------------→
        |       ←   w   →←   w   →←   w   →←   w   → ...
        n       ****************************************
        s
        t       ←-----------------batch----------------→
        e       ←   w   →←   w   →←   w   →←   w   → ...
        p       ****************************************
        |
        |       ←-----------------batch----------------→
        |       ←   w   →←   w   →←   w   →←   w   → ...
        |       ****************************************
        ↓                          ...
        """

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.__config.n_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.__config.n_hidden, forget_bias=1.0)

        outputs, output_state_fw, output_state_bw = \
            tf.nn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                           cell_bw=lstm_bw_cell,
                                           inputs=x,
                                           dtype=tf.float32)

        print("outputs:")  # n_steps * batch_size * (2n_hidden)
        print(outputs)
        print("output_state_fw:")  # batch_size * n_hidden
        print(output_state_fw)
        print("output_state_bw:")  # batch_size * n_hidden
        print(output_state_bw)

        weights = tf.Variable(
            initial_value=tf.random_normal(
                shape=[2 * self.__config.n_hidden, self.__config.n_classes]),
            name="weights")
        biases = tf.Variable(
            initial_value=tf.random_normal(shape=[self.__config.n_classes]),
            name="biases")
        # Calc the logits.
        logits = tf.matmul(outputs[-1], weights) + biases
        # Calc the costs.
        costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y,
            logits=logits))
        # Use adam optimizer to train.
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.__config.learning_rate).minimize(loss=costs)
        # Calc the accuracy.
        correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return costs, train_op, accuracy

    @property
    def costs(self):
        return self.__costs

    @property
    def train_op(self):
        return self.__train_op

    @property
    def accuracy(self):
        return self.__accuracy

    def SGD(self, mnist, max_epoch):
        """Training the Bi-RNN using stochastic gradient descend algorithm.

        Args:
            mnist: The mnist input data.
            max_epoch: The max epoch.

        Returns:
            training_accuracys: The training accuracy of every epoch.
            validation_accuracys: The validation accuracy of every epoch.
            test_accuracys: The test accuracy of every epoch.
        """
        print("SGD")
        session = tf.InteractiveSession()
        writer = tf.summary.FileWriter("./", session.graph)
        tf.global_variables_initializer().run(session=session)

        training_data = mnist.train
        validation_data = mnist.validation
        test_data = mnist.test

        training_data_size = training_data.images.shape[0]
        validation_data_size = validation_data.images.shape[0]
        test_data_size = test_data.images.shape[0]
        # Calc the batches number of training
        num_train_mini_batches = (int)(training_data_size / \
                                       self.__config.batch_size)
        # num_validation_mini_batches = (int)(validation_data.images.shape[0]
        #                                     / \
        #                                     self.__config.batch_size)
        # num_test_mini_batches = (int)(test_data.images.shape[0] / \
        #                               self.__config.batch_size)

        print("num_train_mini_batches:")
        print(num_train_mini_batches)
        # print("num_validation_mini_batches:")
        # print(num_validation_mini_batches)
        # print("num_test_mini_batches:")
        # print(num_test_mini_batches)

        best_validation_accuracy = 0.0
        training_accuracys = []
        validation_accuracys = []
        for epoch in range(max_epoch):
            # print("epoch {0} begin##########################".format(epoch))
            start_time = time.time()
            for i in range(num_train_mini_batches):
                batch_xs, batch_ys = training_data.next_batch(
                    self.__config.batch_size)
                batch_xs = np.reshape(batch_xs, [self.__config.batch_size, 28,
                                                 28])
                self.__train_op.run({self.__x: batch_xs,
                                     self.__y: batch_ys})

            # Calc the training accuracy.
            training_accuracy = self.__accuracy.eval({
                self.__x:
                    np.reshape(
                        training_data.images, [training_data_size, 28, 28]),
                self.__y:
                    training_data.labels})
            # Calc the validation accuracy.
            validation_accuracy = self.__accuracy.eval({
                self.__x:
                    np.reshape(
                        validation_data.images, [validation_data_size, 28, 28]),
                self.__y:
                    validation_data.labels})
            training_accuracys.append(training_accuracy)
            validation_accuracys.append(validation_accuracy)
            time_cost = time.time() - start_time
            print("Epoch {0} use {1:.5} s, training_accuracy: {2:.2%}, "
                  "validation_accuracy: {3:.2%}".format(epoch + 1,
                                                        time_cost,
                                                        training_accuracy,
                                                        validation_accuracy))
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                # Calc the test accuracy.
                test_accuracy = self.__accuracy.eval({
                    self.__x: np.reshape(
                        test_data.images, [test_data_size, 28, 28]),
                    self.__y: test_data.labels})
                print("Get better validation_accuracy, test_accuracy:{"
                      "0:.2%}".format(test_accuracy))
        print(
            "Training finished, the best validation accuracy: {0:.2%} with "
            "test accuracy: {1:.2%}".format(best_validation_accuracy,
                                            test_accuracy))
        writer.close()
        return training_accuracys, validation_accuracys, test_accuracy


def plot_accuracy(training_accuracy, evaluation_accuracy,
                  best_test_accuracy):
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


if __name__ == "__main__":
    bi_rnn = BiRNN(Config)
    mnist = load_mnist_data()
    training_accuracy, evaluation_accuracy, best_test_accuracy = bi_rnn.SGD(
        mnist, 50)
    plot_accuracy(training_accuracy, evaluation_accuracy, best_test_accuracy)
