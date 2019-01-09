#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""The Bi-RNN model to train MNIST data.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/21 9:36
"""
# Common libs.
import time

# 3rd-part libs.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy

def load_mnist_data():
    """Load mnist data.

    :return: mnist data.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist.train, mnist.validation, mnist.test
