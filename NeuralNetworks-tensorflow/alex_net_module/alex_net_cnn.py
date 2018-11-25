#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The AlexNet-CNN used to learn and classify ImageNet datas.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/9 9:38
"""

# System libs
import time
import math

# 3rd part libs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tools.img_net_tf_records_reader as img_net_reader

tf_records_dir = "../image_net_records_files/"


def load_random_image_net_datas(batch_size):
    """Load random images and labels with mini batch size.

    Args:
        batch_size: The mini batch size.

    Returns:
        The data with mini batch size.
    """
    images_train = np.random.normal(size=[batch_size, 224, 224, 3])
    images_test = np.random.normal(size=[batch_size, 224, 224, 3])
    labels_train = np.random.randint(1, 10, batch_size)
    labels_test = np.random.randint(1, 10, batch_size)
    return images_train, labels_train, images_test, labels_test


def load_image_net_datas(batch_size):
    """Load image net data from image net input.

    Args:
        batch_size: The mini batch size.

    Returns:
        Training and test mini batch datas.
    """
    image_train, label_train = img_net_reader.load_distorted_inputs(tf_records_dir,
                                                                    batch_size)
    image_test, label_test = img_net_reader.load_inputs(True, tf_records_dir,
                                                        batch_size)
    return image_train, label_train, image_test, label_test


def describe_tensor(t):
    """Print the description of the one tensor structure.

    Args:
        t: The tensor data.
    """
    print(t.op.name, " ", t.get_shape().as_list())


class Network(object):
    """The AlexNet CNN."""

    def __init__(self, mini_batch):
        """The construct function of AlexNet CNN.

        Args:
            mini_batch: The mini batch size.
        """
        self.mini_batch = mini_batch
        self.eta = tf.placeholder(dtype=tf.float32)

        # Define inputs placeholder.
        self.images = tf.placeholder(dtype=tf.float32, shape=[mini_batch, 224,
                                                              224, 3],
                                     name="images")
        self.labels_ = tf.placeholder(dtype=tf.int32, shape=[mini_batch],
                                      name="labels")
        self.train_op, self.loss, self.top_k_op = self.__inference(self.images,
                                                                   self.labels_)

    def __inference(self, images, labels):
        """Construct all layers of AlexNet CNN and the total loss. Then define
         the optimizer.

        Args:
            images: The input images.
            labels: The labels of these images.

        Returns:
            train_op: The training operation.
            top_k_op: The operation to calc the accuracy.
        """
        parameters = []
        # The first convolution layer.
        with tf.name_scope("conv1") as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 64],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                                 name="weights")
            describe_tensor(kernel)
            bias = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                               name="bias")
            describe_tensor(bias)
            conv = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1],
                                padding="SAME", name="conv")
            describe_tensor(conv)
            conv1 = tf.nn.relu(tf.add(conv, bias), name="output")
            describe_tensor(conv1)
            parameters += [kernel, bias]
        # Define the first local response normalization layer.
        lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9,
                         beta=0.75, name="lrn1")
        describe_tensor(lrn1)
        # Define the first max pool layer.
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID",
                               name="pool1")
        describe_tensor(pool1)

        # Define the second convolution layer.
        with tf.name_scope("conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 192],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                                 name="weights")
            describe_tensor(kernel)
            bias = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                               name="bias")
            describe_tensor(bias)
            conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1],
                                padding="SAME", name="conv")
            describe_tensor(conv)
            conv2 = tf.nn.relu(tf.add(conv, bias), name="output")
            describe_tensor(conv2)
            parameters += [kernel, bias]
        # Define the second local response normalization layer.
        lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9,
                         beta=0.75, name="lrn2")
        describe_tensor(lrn2)
        # Define the second max pool layer.
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID",
                               name="pool2")
        describe_tensor(pool2)

        # Define the third convolution layer.
        with tf.name_scope("conv3") as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 384],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                                 name="weights")
            describe_tensor(kernel)
            bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                               name="bias")
            describe_tensor(bias)
            conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1],
                                padding="SAME", name="conv")
            describe_tensor(conv)
            conv3 = tf.nn.relu(tf.add(conv, bias), name="output")
            describe_tensor(conv3)
            parameters += [kernel, bias]

        # Define the forth convolution layer.
        with tf.name_scope("conv4") as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                                 name="weights")
            describe_tensor(kernel)
            bias = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                name="bias")
            describe_tensor(bias)
            conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1],
                                padding="SAME", name="conv")
            describe_tensor(conv)
            conv4 = tf.nn.relu(tf.add(conv, bias), name="output")
            describe_tensor(conv4)
            parameters += [kernel, bias]

        # Define the fifth convolution layer.
        with tf.name_scope("conv5") as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                                 name="weights")
            describe_tensor(kernel)
            bias = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                name="bias")
            describe_tensor(bias)
            conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1],
                                padding="SAME", name="conv")
            describe_tensor(conv)
            conv5 = tf.nn.relu(tf.add(conv, bias), name="output")
            describe_tensor(conv5)
            parameters += [kernel, bias]
        # Define the fifth max pool layer.
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID",
                               name="pool5")
        describe_tensor(pool5)

        # Define the sixth full-connected layer.
        # Flatten the cnn output.
        flatten = tf.reshape(pool5, shape=[self.mini_batch, -1])
        describe_tensor(flatten)
        with tf.name_scope("full-con6") as scope:
            dim = flatten.get_shape()[1].value
            weights = self.__init_weights_with_loss([dim, 4096],
                                                    stddev=1 / np.sqrt(4096),
                                                    wl=0.004)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(1.0,
                                           dtype=tf.float32,
                                           shape=[4096]),
                               name="bias")
            describe_tensor(bias)
            local6 = tf.nn.relu(tf.add(tf.matmul(flatten, weights), bias),
                                name="output")
            describe_tensor(local6)
            parameters += [weights, bias]

        # Define the seventh full-connected layer.
        with tf.name_scope("full-con7") as scope:
            weights = self.__init_weights_with_loss([4096, 4096],
                                                    stddev=1 / np.sqrt(4096),
                                                    wl=0.004)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(1.0,
                                           dtype=tf.float32,
                                           shape=[4096]),
                               name="bias")
            local7 = tf.nn.relu(tf.add(tf.matmul(local6, weights), bias),
                                name="output")
            describe_tensor(local7)
            parameters += [weights, bias]

        # Define the eighth full-connected layer.
        with tf.name_scope("full-con8") as scope:
            weights = self.__init_weights_with_loss([4096, 1000],
                                                    stddev=1 / np.sqrt(
                                                        1000),
                                                    wl=0.004)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(1.0,
                                           dtype=tf.float32,
                                           shape=[1000]),
                               name="bias")
            local8 = tf.nn.relu(tf.add(tf.matmul(local7, weights), bias),
                                name="output")
            describe_tensor(local8)
            parameters += [weights, bias]

        # Calc the total losses.
        loss = self.__calc_loss(local8, labels)

        # Use adam optimizer.
        train_op = tf.train.AdamOptimizer(self.eta).minimize(
            loss=loss)
        # Define the accuracy calculation.
        top_k_op = tf.nn.in_top_k(local8, labels, 1)
        return train_op, loss, top_k_op

    def __calc_loss(self, logits, labels):
        """Define the loss by logits and desired labels.

        Args:
            logits: The logits output of CNN. Original, unscaled, can be
        regarded as an unnormalized log probability.
            labels: The desired labels.

        Returns:
            The loss of the CNN.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="cross_entropy_per_example")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

        # Add weights losses of normalization into the total losses.
        tf.add_to_collection("losses", cross_entropy_mean)
        return tf.add_n(tf.get_collection("losses"), name="total_losses")

    def __init_weights_with_loss(self, shape, stddev, wl, normal_func="L2"):
        """Initialize the weights with loss of normalization.

        Args:
            shape: The weights shape.
            stddev: The standard deviation.
            wl: The normalization function. It can be "L1" or "L2".
            normal_func:

        Returns:
            The weights initialized with loss of normalization.
        """
        # Use gaussian distribution to initialize the weights.
        weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev,
                                                  dtype=tf.float32),
                              name="weights")

        if wl is not None:
            if normal_func == "L2":
                weight_loss = tf.multiply(tf.nn.l2_loss(weights), wl,
                                          name="weights_loss")
                tf.add_to_collection("losses", weight_loss)
        return weights

    def SGD(self, eta, steps, test_sample_size=1000):
        """Training and test AlexNet CNN using stochastic gradient descent.

        Args:
            eta: The learning rate.
            steps: The total steps to train network.
            test_sample_size: The size of sample to test.

        Returns:
            training_accuracys: The list containing training accuracy in every
            step.
            evaluation_accuracys: The list containing evaluation accuracy in
            every step.

        """
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        images_train, labels_train, images_test, labels_test = \
            load_image_net_datas(self.mini_batch)
        tf.train.start_queue_runners()

        test_iter = int(math.ceil(test_sample_size / self.mini_batch))
        train_accuracys = []
        test_accuracys = []
        for step in range(steps):
            start_time = time.time()
            images_train_data, labels_train_data, images_test_data, \
            labels_test_data = sess.run([
                images_train, labels_train,
                images_test, labels_test])
            train_op, train_loss, train_top_k = \
                sess.run([self.train_op, self.loss, self.top_k_op],
                         feed_dict={self.eta: eta,
                                    self.images: images_train_data,
                                    self.labels_: labels_train_data})
            train_accuracy = np.sum(train_top_k) / self.mini_batch
            train_accuracys.append(train_accuracy)

            test_accuracy = 0.0
            for i in range(test_iter):
                test_accuracy = test_accuracy + np.sum(
                    sess.run([self.top_k_op],
                             feed_dict=
                             {self.eta: eta,
                              self.images: images_test_data,
                              self.labels_: labels_test_data})) / \
                                self.mini_batch

            test_accuracy = test_accuracy / test_iter
            test_accuracys.append(test_accuracy)
            time_cost = time.time() - start_time
            if step % 10 == 0:
                print("Step({0}) end with {1}s".format(step, time_cost))
            print(
                "train_loss:{0:.5}, train_accuracy:{1:.2%}, validation_accuracy:{"
                "2:.2%}".format(train_loss, train_accuracy, test_accuracy))
        return train_accuracys, test_accuracys


def plot_accuracy(training_accuracy, evaluation_accuracy):
    """Plot the accuracy change of all epochs.

    Args:
        training_accuracy: The training accuracy list.
        evaluation_accuracy: The evaluation accuracy list.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, len(training_accuracy), 1)
    ax.plot(x, training_accuracy, label='train accuracy')
    ax.plot(x, evaluation_accuracy, label='validation accuracy')
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("The best validation accuracy:{0:.2%}".format(
        np.max(evaluation_accuracy)))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    network = Network(10)
    training_accuracys, validation_accuracys = \
        network.SGD(eta=1e-3,
                    steps=500)
    plot_accuracy(training_accuracys, validation_accuracys)
