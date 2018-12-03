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
import alex_net_module.cifar10_input_for_alex_net as cifar10_input
from PIL import Image
import alex_net_module.constants as constants

data_dir = "/tmp/cifar10_data/cifar-10-batches-bin"

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
    # Load the ImageNet data.
    # image_train, label_train = img_net_reader.load_distorted_inputs(
    #                                                       constants.tf_records_dir,
    #                                                       batch_size)
    # image_test, label_test = img_net_reader.load_inputs(True, constants.tf_records_dir,
    #                                                     batch_size)

    # Load the cifar10 data.
    image_train, label_train = cifar10_input.distorted_inputs(data_dir, batch_size)
    image_test, label_test = cifar10_input.inputs(True, data_dir, batch_size)
    return image_train, label_train, image_test, label_test


def describe_tensor(t):
    """Print the description of the one tensor structure.

    Args:
        t: The tensor data.
    """
    print(t.op.name, " ", t.get_shape().as_list())


class Network(object):
    """The AlexNet CNN."""

    def __init__(self, mini_batch, keep_prob):
        """The construct function of AlexNet CNN.

        Args:
            mini_batch: The mini batch size.
            keep_prob: The dropout keep probability.
        """
        self.mini_batch = mini_batch
        self.keep_prob = keep_prob
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
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias), name="output")
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
            conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias), name="output")
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
            conv3 = tf.nn.relu(tf.nn.bias_add(conv, bias), name="output")
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
            conv4 = tf.nn.relu(tf.nn.bias_add(conv, bias), name="output")
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
            conv5 = tf.nn.relu(tf.nn.bias_add(conv, bias), name="output")
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
                                                    stddev=0.01,
                                                    wl=0.008)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(0.1,
                                           dtype=tf.float32,
                                           shape=[4096]),
                               name="bias")
            describe_tensor(bias)
            local6 = tf.nn.relu(tf.matmul(flatten, weights) + bias,
                                name="output")
            describe_tensor(local6)
            dropout6 = tf.nn.dropout(local6, keep_prob=self.keep_prob)
            parameters += [weights, bias]

        # Define the seventh full-connected layer.
        with tf.name_scope("full-con7") as scope:
            weights = self.__init_weights_with_loss([4096, 4096],
                                                    stddev=0.01,
                                                    wl=0.008)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(0.1,
                                           dtype=tf.float32,
                                           shape=[4096]),
                               name="bias")
            local7 = tf.nn.relu(tf.matmul(local6, weights) + bias,
                                name="output")
            describe_tensor(local7)
            dropout7 = tf.nn.dropout(local7, keep_prob=self.keep_prob)
            parameters += [weights, bias]

        # Define the eighth full-connected layer.
        with tf.name_scope("full-con8") as scope:
            weights = self.__init_weights_with_loss([4096, 10],
                                                    stddev=0.01,
                                                    wl=0.0)
            describe_tensor(weights)
            bias = tf.Variable(tf.constant(0.0,
                                           dtype=tf.float32,
                                           shape=[10]),
                               name="bias")
            local8 = tf.matmul(local7, weights) + bias
            describe_tensor(local8)
            # logits = tf.nn.sigmoid(local8, "logits")
            # describe_tensor(logits)

            parameters += [weights, bias]

        # Calc the total losses.
        loss = self.__calc_loss(local8, labels)

        # Use adam optimizer.
        train_op = tf.train.AdamOptimizer(self.eta).minimize(
            loss=loss)
        # Define the accuracy calculation.
        top_k_op = tf.nn.in_top_k(local8, labels, 1)
        self.pred = tf.nn.softmax(local8)
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
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="cross_entropy_per_example")
        describe_tensor(cross_entropy)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
        describe_tensor(cross_entropy_mean)

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

    def SGD(self, eta, epochs, epoch_train_size, test_sample_size=1000):
        """Training and test AlexNet CNN using stochastic gradient descent.

        Args:
            eta: The learning rate.
            epochs: Thr train epochs.
            epoch_train_size: The train size of every epoch.
            test_sample_size: The size of sample to test.

        Returns:
            training_accuracys: The list containing training accuracy in every
            step.
            evaluation_accuracys: The list containing evaluation accuracy in
            every step.

        """
        # Calc the steps of every epoch.
        every_epoch_steps = epoch_train_size / self.mini_batch
        # Calc the total steps of total epochs.
        steps = int(epochs * every_epoch_steps)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        images_train, labels_train, images_test, labels_test = \
            load_image_net_datas(self.mini_batch)
        tf.train.start_queue_runners()

        test_iter = int(math.ceil(test_sample_size / self.mini_batch))
        test_accuracys = []

        for step in range(steps):
            start_time = time.time()
            images_train_data, labels_train_data = sess.run([
                images_train, labels_train])
            train_op, train_loss = \
                sess.run([self.train_op, self.loss],
                         feed_dict={self.eta: eta,
                                    self.images: images_train_data,

                                    self.labels_: labels_train_data})
            if step % 1000 == 0:
                time_cost = time.time() - start_time
                print("Train step({0}) end with {1}s with train_loss:{"
                      "2}".format(step, time_cost, train_loss))

            if step % every_epoch_steps == 0 or step == steps - 1: # One epoch.
                test_accuracy = 0.0
                for i in range(test_iter):
                    images_test_data, labels_test_data = sess.run([
                        images_test, labels_test])
                    test_loss, top_k_op= \
                        sess.run([self.loss, self.top_k_op],
                                 feed_dict={self.eta: eta,
                                            self.images: images_test_data,
                                            self.labels_: labels_test_data})
                    # print(np.sum(top_k_op))
                    test_accuracy = test_accuracy + np.sum(top_k_op) / \
                                    float(self.mini_batch)

                test_accuracy = float(test_accuracy) / float(test_iter)
                test_accuracys.append(test_accuracy)
                print(
                    "test_loss:{0:.5}, test_accuracy:{1:.2%}"
                        .format(test_loss, test_accuracy))

                # If accuracy don't upgrade after every 10 epochs. Update
                # eta: Eta = eta / 10.
                if step != 0 and step % (every_epoch_steps * 10) == 0:
                    if test_accuracys[int(step / every_epoch_steps - 10)] >=\
                            test_accuracy:
                        eta = eta / 10
                        print("Accuracy not upgrade after 10 epochs. Adjust "
                              "the eta. Now eta:{0}".format(eta))

        return test_accuracys

    def predict(self, image_file):
        try:
            img = Image.open(image_file)
        except OSError as e:
            print(e)
            print("Error image " + image_file)
            return

        # Unify resolution to 224 * 224.
        img = np.array(
            img.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE)))
        # print("img:")
        # print(img)

        # Check if the image is rgb image.
        if len(img.shape) != 3 or img.shape[2] != 3:
            print("Not rgb image " + image_file)
            return

        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.cast(img_tensor, tf.float32)
        # Standardize the image data.
        stand_img_tensor = tf.image.per_image_standardization(img_tensor)
        stand_img_tensor.set_shape([constants.IMAGE_SIZE, constants.IMAGE_SIZE,
                                    3])

        sess = tf.Session()
        stand_img = np.array(sess.run([stand_img_tensor]))
        # print("stand_img:")
        # print(stand_img)
        # print(np.mean(stand_img[0]))
        # print(np.var(stand_img[0]))

        # Load model
        model_saver = tf.train.Saver()
        model_saver.restore(sess, "../model_saver/alex_net_slp-190000")
        predict = sess.run([self.pred], feed_dict={self.images: stand_img})
        print("predict:")
        print(predict[0])
        print("I guess this image is {0}".format(constants.labels[np.argmax(
            predict)]))


def plot_accuracy(evaluation_accuracy):
    """Plot the accuracy change of all epochs.

    Args:
        evaluation_accuracy: The evaluation accuracy list.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, len(evaluation_accuracy), 1)
    ax.plot(x, evaluation_accuracy, label='validation accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("The best validation accuracy:{0:.2%}".format(
        np.max(evaluation_accuracy)))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Train alex net using cifar10 data.
    network = Network(mini_batch=20, keep_prob=0.6)
    validation_accuracys = \
        network.SGD(eta=0.001,
                    epochs=200,
                    epoch_train_size=40000,
                    test_sample_size=5000)
    plot_accuracy(validation_accuracys)
    sess = tf.InteractiveSession()
    model_saver = tf.train.Saver()
    model_saver.save(sess, "../model_saver/alex_net.ckpt",
                     global_step=(200 * 40000))

    # Train alex net using ImageNet data.
    """
    network = Network(mini_batch=10, keep_prob=0.6)
    validation_accuracys = \
        network.SGD(eta=0.001,
                    epochs=200,
                    epoch_train_size=10000,
                    test_sample_size=1000)
    plot_accuracy(validation_accuracys)
    sess = tf.InteractiveSession()
    model_saver = tf.train.Saver()
    model_saver.save(sess, "../model_saver/alex_net.ckpt",
                     global_step=(200 * 10000))
    network.predict("../image_net_origin_files/dog/dog_419.jpg")
    """
