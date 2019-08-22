#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
Construct the auto encoder including gaussian-noise encoder, noiseless encoder
and random mask noise encoder.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/10/26 9:22
"""

# The 3rd part libs
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    """The xavier init function used to initialize the params.

    :param fan_in: The number of input.
    :param fan_out: The number of output.
    :param constant: The const.
    :return: The xavier initialization of params.
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    # use uniform distribution.
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):
    """The additive gaussian noise auto encoder.

    Attributes:
        n_input: The number of inputs.
        n_hidden: The number of hidden layers.
        act_func: The activation function.
        optimizer: The optimizer.
        scale: The gaussian noise param.
    """

    def __init__(self, n_input, n_hidden, act_func=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """The constructor function of additive gaussian noise auto encoder."""
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.act_func = act_func
        self.optimizer = optimizer
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self.__initialize_weights()
        self.weights = network_weights

        # Define the input layer.
        self.x = tf.placeholder(tf.float32)
        # Define the hidden layer.
        self.hidden_layer = self.act_func(tf.add(
            tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                      self.weights['w1']), self.weights['b1']))
        # Define the output layer.
        self.reconstruction = tf.add(tf.matmul(self.hidden_layer,
                                               self.weights['w2']),
                                     self.weights['b2'])
        # Define the cost.
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,
                                                     self.x), 2.0))
        # Define the optimizer.
        self.optimizer = optimizer.minimize(self.cost)
        # Init the tf global variables.
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def __initialize_weights(self):
        """Initialize all weights.

        :return: The initial weights.
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input],
                                                 dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    def train(self, x_train, x_test, epoches=20, batch_size=128,
              display_setp=1):
        """Train the network by inputs.

        :param x_train: The training x.
        :param x_test: The test x.
        :param epoches: The epoches.
        :param batch_size: The batch size.
        :param display_setp: The step of display message.

        :return:
            min_cost: The minimum cost.
        """

        def partial_fit(x):
            """The training function."""
            cost, opt = self.session.run((self.cost, self.optimizer),
                                         feed_dict={self.x: x, self.scale:
                                             self.training_scale})
            return cost

        train_batches = int(x_train.shape[0] / batch_size)

        for epoch in range(epoches):
            avg_cost = 0.0
            for i in range(train_batches):
                x_batch = self.get_random_block_from_data(x_train, batch_size)
                cost = partial_fit(x_batch)
                avg_cost += cost / train_batches
            test_cost = partial_fit(x_test)
            if epoch % display_setp == 0:
                print("The epoch({0}): train average cost:{1:.9f}, test cost:{"
                      "2:.9f}".format(epoch, avg_cost, test_cost))

    def get_random_block_from_data(self, x, batch_size):
        """Return random mini batch.

        :param x: The inputs
        :param batch_size: The mini batch size.
        :return: The mini batch.
        """
        start_index = np.random.randint(0, x.shape[0] - batch_size)
        return x[start_index: start_index + batch_size]

    def calc_total_cost(self, x):
        """Calculate the total cost.

        :param x: The inputs.
        :return: The total cost.
        """
        return self.session.run(self.cost, feed_dict={self.x: x, self.scale:
            self.training_scale})

    def transform(self, x):
        """Transform the inputs to high dimension characteristic.

        :param x: The inputs.
        :return: The high dimension characteristic.
        """
        return self.session.run(self.hidden_layer,
                                feed_dict={self.x: x, self.scale:
                                    self.training_scale})

    def generate(self, hidden_layer):
        """Generate the original data by hidden layer.

        :param hidden_layer: The hidden layer.
        :return: The original inputs data.
        """
        return self.session.run(self.reconstruction,
                                feed_dict={self.hidden_layer: hidden_layer})

    def reconstruct(self, x):
        """The combination of transform with generate.

        :param x: The inputs.
        :return: The inputs reconstructed.
        """
        return self.session.run(self.reconstruction,
                                feed_dict={self.x: x, self.scale:
                                    self.training_scale})

    def standard_data(self, x_train, x_test):
        """Standard the training data and test data. If the average of data
        is u and standard deviation of data is o, the standard data is (x -
        u) / o.

        :param x_train: The training data.
        :param x_test: The test data.
        :return: The standard training data and test data.
        """
        preprocessor = prep.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)
        return x_train, x_test

if __name__=="__main__":
    autoEncoder = AdditiveGaussianNoiseAutoEncoder(
        n_input=784,
        n_hidden=200,
        act_func=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        scale=0.01
    )
    mnist = input_data.read_data_sets("datas/MNIST_data/", one_hot=True)
    x_train, x_test = autoEncoder.standard_data(mnist.train.images,
                                                mnist.test.images)
    print("x_train shape:")
    print(x_train.shape)
    print("x_test shape:")
    print(x_test.shape)
    # x_train, x_test = mnist.train.images, mnist.test.images
    autoEncoder.train(
        x_train=x_train,
        x_test=x_test,
        epoches=20,
        batch_size=128,
        display_setp=1
    )

