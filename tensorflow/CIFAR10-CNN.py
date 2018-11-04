#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
A CNN instance to classify CIFAR-10.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/10/31 19:39
"""

# 3rd lib.
import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

data_dir = "/tmp/cifar10_data/cifar-10-batches-bin"


def load_cifar10_datas(batch_size):
    """

    :param batch_size:
    :return:
    """
    cifar10.maybe_download_and_extract()
    # Return train data has been augmented.
    images_train, labels_train = cifar10_input.inputs(False, data_dir,
                                                      batch_size)
    # Return the validation data.
    images_test, labels_test = cifar10_input.inputs(True, data_dir, batch_size)
    return images_train, labels_train, images_test, labels_test


class Network(object):
    """
    Main class to construct and train network.
    """

    def __init__(self, mini_batch):
        """The construct function of network.

        :param mini_batch: The mini batch.
        """
        self.mini_batch = mini_batch
        self.eta = tf.placeholder(dtype=tf.float32)

        # Define the images input and desired label output as place holder.
        self.images = tf.placeholder(dtype=tf.float32, shape=[
            self.mini_batch, 32, 32, 3])
        self.labels_ = tf.placeholder(dtype=tf.int32, shape=[
            self.mini_batch])

        # Define the first CNN layer.
        self.weights1 = self.__init_weights_with_loss([5, 5, 3, 64],
                                                      stddev=5e-2, wl=0.0)
        self.kernel1 = tf.nn.conv2d(self.images, self.weights1,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME")
        self.bias1 = tf.Variable(tf.zeros([64]), dtype=tf.float32)
        self.conv1 = tf.nn.relu(tf.nn.bias_add(self.kernel1, self.bias1))
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding="SAME")
        # Local response normalization.
        self.norm1 = tf.nn.lrn(input=self.pool1,
                               depth_radius=4,
                               bias=1.0,
                               alpha=0.001 / 9.0,
                               beta=0.75)

        # Define the Second CNN layer.
        self.weights2 = self.__init_weights_with_loss([5, 5, 64, 64],
                                                      stddev=5e-2, wl=0.0)
        self.kernel2 = tf.nn.conv2d(self.norm1, self.weights2,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME")
        self.bias2 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[64]))
        self.conv2 = tf.nn.relu(tf.nn.bias_add(self.kernel2, self.bias2))
        self.norm2 = tf.nn.lrn(input=self.conv2,
                               depth_radius=4,
                               bias=1.0,
                               alpha=0.001 / 9.0,
                               beta=0.75)
        self.pool2 = tf.nn.max_pool(self.norm2, ksize=[1, 3, 3, 1], strides=[
            1, 2, 2, 1], padding="SAME")

        # Define the first full-connected layer.
        # Flatten the cnn output.
        reshape = tf.reshape(self.pool2, [self.mini_batch, -1])
        dim = reshape.get_shape()[1].value
        self.weights3 = self.__init_weights_with_loss([dim, 384],
                                                      stddev=0.04,
                                                      wl=0.004)
        self.bias3 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[
            384]))
        self.local3 = tf.nn.relu(tf.matmul(reshape, self.weights3) + self.bias3)

        # Define the second full-connected layer.
        self.weights4 = self.__init_weights_with_loss(shape=[384, 192],
                                                      stddev=0.04,
                                                      wl=0.004)
        self.bias4 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[
            192]))
        self.local4 = tf.nn.relu(tf.matmul(self.local3, self.weights4) +
                                 self.bias4)

        # Define the output layer.
        self.weights5 = self.__init_weights_with_loss(shape=[192, 10],
                                                      stddev=1 / 192, wl=0.0)
        self.bias5 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[10]))
        self.logits = tf.add(tf.matmul(self.local4, self.weights5), self.bias5)

        # Layer define finish.
        # The all layers is described as follow:
        # conv1
        # max_pool1
        # lrn1
        # conv2
        # lrn2
        # max_pool2
        # full_connected1
        # full_connected2
        # logits

        # Calc the total losses.
        self.loss = self.__calc_loss(self.logits, self.labels_)

        # Use adam optimizer.
        self.train_op = tf.train.AdamOptimizer(self.eta).minimize(
            loss=self.loss)
        self.top_k_op = tf.nn.in_top_k(self.logits, self.labels_, 1)

    def SGD(self, eta=1e-3, steps=3000):
        """Train the network using mini-batch stochastic gradient descent.

        :param eta: The learning rate.
        :param steps: The training steps.
        :return:
            training_accuracy: The list containing training accuracy in every
           epoch.
            evaluation_accuracy: The list containing evaluation accuracy in
           every epoch.
        """
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Start thread queue for data augmentation.
        # tf.train.start_queue_runners()

        images_train, labels_train, images_test, labels_test = \
        load_cifar10_datas(self.mini_batch)

        train_accuracys = []
        test_accuracys = []
        for step in range(steps):
            start_time = time.time()
            images_train_batch, labels_train_batch = sess.run([images_train,
                                                               labels_train])
            images_test_batch, labels_test_batch = sess.run([images_test,
                                                             labels_test])
            train_op, train_loss, train_accuracy = \
                sess.run([self.train_op, self.loss, self.top_k_op],
                         {self.mini_batch: self.mini_batch,
                          self.eta: eta,
                          self.images: images_train_batch,
                          self.labels_: labels_train_batch})
            train_accuracys.append(train_accuracy)
            test_accuracy = \
                sess.run([self.top_k_op],
                         {self.mini_batch: self.mini_batch,
                          self.eta: eta,
                          self.images: images_test_batch,
                          self.labels_: labels_test_batch})
            test_accuracys.append(test_accuracy)
            time_cost = time.time() - start_time
            if step % 10 == 0:
                print("Step({0}) end with {1}s".format(step, time_cost))
                print("train_accuracy:{0:.2%}, validation_accuracy:{"
                      "1:.2%}".format(train_accuracy, test_accuracy))
        return train_accuracys, test_accuracys

    def __calc_loss(self, logits, labels):
        """Define the loss by logits and desired labels.

        :param logits: The logits output of CNN. Original, unscaled, can be
        regarded as an unnormalized log probability.
        :param labels: The desired labels.
        :return: The loss of the CNN.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="cross_entropy_per_example")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

        # Add weights losses of normalization into the total losses.
        tf.add_to_collection("losses", cross_entropy_mean)
        return tf.add_n(tf.get_collection("losses"), name="total_losses")

    def __init_weights_with_loss(self, shape, stddev, wl, normal_func="L2"):
        """Initialize the weights with loss of normalization.

        :param shape: The weights shape.
        :param stddev: The standard deviation.
        :param wl: The normalization params.
        :param normal_func: The normalization function. It can be "L1" or "L2".
        :return: The weights initialized with loss of normalization.
        """
        # Use gaussian distribution to initialize the weights.
        weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev))

        if wl is not None:
            if normal_func == "L2":
                weight_loss = tf.multiply(tf.nn.l2_loss(weights), wl,
                                          name="weights_loss")
                tf.add_to_collection("losses", weight_loss)
        return weights


def plot_accuracy(training_accuracy, evaluation_accuracy):
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
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("The final validation accuracy:{0:.2%}".format(
        evaluation_accuracy[-1]))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # images_train, label_train, images_test, label_test = load_cifar10_datas(280)
    # print("images_train shape:")
    # print(images_train.shape)
    # print("label_train shape:")
    # print(label_train.shape)
    # print("images_test shape:")
    # print(images_test.shape)
    # print("label_test shape:")
    # print(label_test.shape)
    network = Network(mini_batch=280)
    train_accuracys, validation_accuracys = network.SGD(eta=1e-3, steps=3000)
    plot_accuracy(train_accuracys, validation_accuracys)