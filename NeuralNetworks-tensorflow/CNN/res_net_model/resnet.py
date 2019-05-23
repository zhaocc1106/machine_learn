#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a resnet model to classify cifar10 data or imagenet data.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/5/19 下午6:31
"""

# common libs.
import os
import time
import pprint
import math

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets
import CNN.cifar10_model.cifar10 as cifar10
import CNN.res_net_model.cifar10_input_for_resnet as cifar10_input
import matplotlib.pyplot as plt

data_dir = "/tmp/cifar10_data/cifar-10-batches-bin"
pre_trained_ckpt = "../../model_saver/resnet_model/pre-trained_model/resnet_v1_101.ckpt"


def load_image_datas(batch_size):
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
    cifar10.maybe_download_and_extract()
    image_train, label_train = cifar10_input.distorted_inputs(data_dir,
                                                              batch_size)
    image_test, label_test = cifar10_input.inputs(True, data_dir, batch_size)
    return image_train, label_train, image_test, label_test


def variable_summaries(vars, name):
    """Construct summaries of vars.

    Args:
        vars: The vars.
        name: The name
    """
    tf.summary.histogram(name=name, values=vars)


class Network(object):
    """The resnet model."""

    def __init__(self, arch):
        """The construct function of resnet model.

        Args:
            arch: The architecture. Such as ```50```, ```101```, ```150```, ```200```.
        """
        self.arch = arch

    def train(self, mini_batch, input_func, eta, epochs, epoch_train_size,
              test_sample_size=1000, pre_trained_ckpt=None):
        """Train the model.

        Args:
            mini_batch: The mini batch size.
            eta: The initial learning rate.
            epochs: Thr train epochs.
            epoch_train_size: The train size of every epoch.
            test_sample_size: The size of sample to test.

        Returns:

        """
        self.mini_batch = mini_batch
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[mini_batch, 224, 224, 3],
                                     name="images")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[mini_batch],
                                     name="labels")
        self.eta = tf.placeholder(dtype=tf.float32, shape=[], name="eta")
        self.loss, self.train_op, self.top_k_op = self.__build_network(True)
        return self.__SGD(input_func, eta, epochs, epoch_train_size,
                          test_sample_size, pre_trained_ckpt)

    def __build_network(self, is_training):
        """Build the resnet network model.

        Args:
            is_training: If is training.

        Returns:
            training operation and top 1 prediction operation.
        """

        if self.arch == "101":
            with slim.arg_scope(slim_nets.resnet_v1.resnet_arg_scope()):
                net, end_points = slim_nets.resnet_v1.resnet_v1_101(
                    inputs=self.images,
                    num_classes=10,
                    is_training=is_training,
                )
        print("net:")
        pprint.pprint(net)
        print("end_points:")
        pprint.pprint(end_points)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=tf.reshape(net, shape=[self.mini_batch, 10]),
            name="cross_entropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy")
        # Use adam optimizer.
        train_op = tf.train.MomentumOptimizer(self.eta, 0.9).minimize(
            loss=loss)
        top_k_op = tf.nn.in_top_k(tf.reshape(net, shape=[self.mini_batch, 10]),
                                  self.labels, 1)
        return loss, train_op, top_k_op

    def __SGD(self, input_func, eta, epochs, epoch_train_size,
              test_sample_size=1000, pre_trained_ckpt=None):
        """Training and test resnet model using stochastic gradient descent.

        Args:
            input_func: The load data function.
            eta: The initial learning rate.
            epochs: Thr train epochs.
            epoch_train_size: The train size of every epoch.
            test_sample_size: The size of sample to test.
            pre_trained_ckpt: The pre-trained model checkpoint.

        Returns:
            The list containing evaluation accuracy in every step.

        """
        sess = tf.InteractiveSession()

        # Collect all tensorboard summaries.
        trainable_vars = tf.trainable_variables()
        for vars in trainable_vars:
            variable_summaries(vars, vars.name)
        # Merge all vars suammary.
        merged = tf.summary.merge_all()
        # Prepare accuracy summary.
        test_accuracy_tensor = tf.get_variable(name="test_accuracy",
                                               shape=[],
                                               dtype=tf.float32,
                                               trainable=False)
        accuracy_summary = tf.summary.scalar("test_accuracy",
                                             test_accuracy_tensor)

        tf.global_variables_initializer().run()

        # Create tensorboard writer.
        writer = tf.summary.FileWriter("./", sess.graph)

        # model saver.
        saver_path = os.path.join("../../model_saver/resnet_model", "res" +
                                  self.arch,
                                  time.strftime('%Y-%m-%d-%H-%M'))
        if not os.path.exists(saver_path):
            os.makedirs(saver_path)
        model_saver = tf.train.Saver()

        # load previous checkpoint.
        if pre_trained_ckpt is not None:
            model_saver.restore(sess=sess, save_path=pre_trained_ckpt)

        # Calc the steps of every epoch.
        every_epoch_steps = epoch_train_size / self.mini_batch
        # Calc the total steps of total epochs.
        steps = int(epochs * every_epoch_steps)

        images_train, labels_train, images_test, labels_test = \
            input_func(self.mini_batch)
        tf.train.start_queue_runners()

        test_iter = int(math.ceil(test_sample_size / self.mini_batch))
        test_accuracys = []

        for step in range(1, steps + 1):
            start_time = time.time()
            images_train_data, labels_train_data = sess.run([
                images_train, labels_train])
            train_op, train_loss = \
                sess.run([self.train_op, self.loss],
                         feed_dict={self.eta: eta,
                                    self.images: images_train_data,

                                    self.labels: labels_train_data})
            if step % 1000 == 0:
                time_cost = time.time() - start_time
                print("Train step({0}) with {1:.2}s per step, with train_loss:"
                      "{2:.5}".format(step, time_cost, train_loss))

            # One epoch.
            if step % every_epoch_steps == 0:
                test_accuracy = 0.0
                test_loss_mean = 0.0
                for i in range(test_iter):
                    images_test_data, labels_test_data = sess.run([
                        images_test, labels_test])
                    test_loss, top_k_op = \
                        sess.run([self.loss, self.top_k_op],
                                 feed_dict={self.eta: eta,
                                            self.images: images_test_data,
                                            self.labels: labels_test_data})
                    # print(np.sum(top_k_op))
                    test_accuracy = test_accuracy + np.sum(top_k_op) / \
                                    float(self.mini_batch)
                    test_loss_mean += test_loss

                test_accuracy = test_accuracy / test_iter
                test_loss_mean = test_loss_mean / test_iter
                test_accuracys.append(test_accuracy)
                print("{0} Epoch:{1}, step:{2}, test_loss:{3:.5},"
                      " test_accuracy:{4:.2%}"
                      .format(time.strftime('[%Y-%m-%d %H:%M:%S]'),
                              step / every_epoch_steps,
                              step,
                              test_loss_mean,
                              test_accuracy))

                # If accuracy don't upgrade after every 10 epochs. Update
                # eta: Eta = eta / 10.
                if step != 0 and step != 10 and step % (every_epoch_steps *
                                                        10) == 0:
                    if test_accuracys[int(step / every_epoch_steps - 10) - 1] \
                            >= test_accuracy:
                        eta = eta / 10
                        print("Accuracy not upgrade after 10 epochs. Adjust "
                              "the eta. Now eta:{0}".format(eta))

                # save model saver ckpt.
                model_saver.save(sess, saver_path + "/resnet.ckpt", step)

                # save tensorboard summary.
                run_metadata = tf.RunMetadata()
                writer.add_run_metadata(run_metadata, "step%d" %
                                        step)
                summary = sess.run(merged)
                writer.add_summary(summary, step)
                _ = sess.run(test_accuracy_tensor.assign(test_accuracy))
                summary = sess.run(accuracy_summary)
                writer.add_summary(summary, step)

        writer.close()
        return test_accuracys


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
    network = Network("101")
    test_accuracys = network.train(mini_batch=20,
                                   input_func=load_image_datas,
                                   eta=1e-8,
                                   epochs=50,
                                   epoch_train_size=40000,
                                   test_sample_size=1000,
                                   pre_trained_ckpt="../../model_saver/resnet_model/res101/2019-05-21-21-34/resnet.ckpt-300000")

    plot_accuracy(test_accuracys)
