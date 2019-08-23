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
import math

# 3rd-part libs.
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets
import CNN.cifar10_model.cifar10 as cifar10
import CNN.res_net_model.cifar10_input_for_resnet as cifar10_input
import CNN.res_net_model.constants as constants
import matplotlib.pyplot as plt
import tools.img_net_tf_records_reader as img_net_reader

data_dir = "/tmp/cifar10_data/cifar-10-batches-bin"
# wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
pre_trained_ckpt = "../../out/model_saver/resnet_model/pre-trained_model/resnet_v1_101.ckpt"


def load_image_datas(batch_size, data_type="cifar10"):
    """Load image net data from image net input.

    Args:
        batch_size: The mini batch size.
        data_type: ```cifar10``` or ```imagenet```.

    Returns:
        Training and test mini batch datas.
    """
    if data_type == "imagenet":
        # Load the ImageNet data. 5898 training images and 644 test images.
        # Execute tools.image_net_downloader.py and
        # tools.img_net_tf_records_writer.py before use it.
        image_train, label_train = img_net_reader.load_distorted_inputs(
            constants.tf_records_dir,
            batch_size)
        image_test, label_test = img_net_reader.load_inputs(True,
                                                            constants.tf_records_dir,
                                                            batch_size)
    elif data_type == "cifar10":
        # Load the cifar10 data.
        cifar10.maybe_download_and_extract()
        image_train, label_train = cifar10_input.distorted_inputs(data_dir,
                                                                  batch_size)
        image_test, label_test = cifar10_input.inputs(True, data_dir,
                                                      batch_size)
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
        # Add resnet layer.
        if self.arch == "101":
            with slim.arg_scope(slim_nets.resnet_v1.resnet_arg_scope()):
                net, end_points = slim_nets.resnet_v1.resnet_v1_101(
                    inputs=self.images,
                    # num_classes=10,
                    is_training=is_training,
                )

        # Add full connected layer.
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        cls_score = slim.fully_connected(net,
                                         10,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None,
                                         scope='cls_score')

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=tf.reshape(cls_score, shape=[self.mini_batch, 10]),
            name="cross_entropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy")
        # Use adam optimizer.
        train_op = tf.train.MomentumOptimizer(self.eta, 0.9).minimize(
            loss=loss)
        top_k_op = tf.nn.in_top_k(
            tf.reshape(cls_score, shape=[self.mini_batch, 10]),
            self.labels, 1)
        return loss, train_op, top_k_op

    def __model_restore(self, sess, pre_trained_ckpt):
        """Restore pre-trained model.

        Args:
            sess: The session.
            pre_trained_ckpt: The pre-trained model path.
        """
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(pre_trained_ckpt)
            var_to_shape_map = reader.get_variable_to_shape_map()
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print(
                    "It's likely that your checkpoint file has been compressed "
                    "with SNAPPY.")
            return

        if var_to_shape_map is not None and len(var_to_shape_map) != 0:
            print('Loading initial model weights from {:s}'.format(
                pre_trained_ckpt))
            variables = tf.global_variables()
            # Get variables can restored.
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_to_shape_map:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)

            # Restore variables.
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, pre_trained_ckpt)
            print("Model loaded.")

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

        # Initialize all variables firstly.
        tf.global_variables_initializer().run()

        # Create tensorboard writer.
        tensorboad_path = os.path.join("../../tensorboard/resnet_model",
                                       "res" + self.arch,
                                       time.strftime('%Y-%m-%d-%H-%M'))
        if  not os.path.exists(tensorboad_path):
            os.makedirs(tensorboad_path)
        writer = tf.summary.FileWriter(tensorboad_path, sess.graph)

        # model saver.
        saver_path = os.path.join("../../out/model_saver/resnet_model", "res" +
                                  self.arch,
                                  time.strftime('%Y-%m-%d-%H-%M'))
        if not os.path.exists(saver_path):
            os.makedirs(saver_path)
        model_saver = tf.train.Saver()

        # load previous checkpoint if has pre-trained model.
        if pre_trained_ckpt is not None:
            self.__model_restore(sess, pre_trained_ckpt)

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
                                   eta=1e-3,
                                   epochs=150,
                                   epoch_train_size=40000,
                                   test_sample_size=1000,
                                   # pre_trained_ckpt=pre_trained_ckpt
                                   )
    plot_accuracy(test_accuracys)
