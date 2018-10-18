#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The Convolution Neural Networks Algorithm realized by theano.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/29 19:48
"""
# The common library
import gzip
import cPickle

# Third-party library
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax
from theano.tensor.signal import downsample

#### Constants
GPU = False
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify " + \
          "network3.py\nto set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
    except:
        print("exception")
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify " + \
          "network3.py to set\nthe GPU flag to True."


def ReLu(z):
    """The ReLu function

    Args:
        z: The input of activation function.

    Returns:
        The activation.
    """
    return T.maximum(0.0, z)


def load_mnist_data_shared():
    """Load mnist data and covert to theano.tensor.sharedvar data.

    Returns:
        The training, validation and test data.
    """
    f = gzip.open("mnistData/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    def shared(data):
        """Convert data to theano.tensor.sharedvar var.

        Args:
            data: The data.

        Returns:
            The theano.tensor.sharedvar var.
        """
        shared_x = theano.shared(np.asarray(data[0],
                                            dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(data[1],
                                            dtype=theano.config.floatX),
                                 borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]


### Main class to construct and train networks.
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Use the list of ``layers`` and mini_batch_size to construct the
        network.

        Args:
            layers: The list of ``layers`` composed of the combination of
            convolution layer and full-connected layer and softmax layer.
            mini_batch_size: To describe the mini batch size used during
            training by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in layers for param in layer.params]
        self.x = T.matrix("x")  # define input using theano matrix.
        self.y = T.ivector("y")  # define output using theano vector.

        # Begin construct the symbolic mathematical computation of output
        # with input by theano graphs.
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in range(1, len(self.layers)):
            pre_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(pre_layer.output, pre_layer.output_dropout,
                           self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using stochastic gradient decent.

        Args:
            training_data: The training data.
            epochs: The epochs to train.
            mini_batch_size: The mini batch size.
            eta: The learn rate.
            validation_data: The validation data.
            test_data: The test data.
            lmbda: The regularization param lambda.
        """
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size

        # define the (regularized) cost function
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(
            self) + 0.5 * lmbda * l2_norm_squared / size(training_data)
        # define the grads
        grads = T.grad(cost, self.params)
        # define the updates
        updates = [(param, param - eta * grad) for param, grad in
                   zip(self.params, grads)]

        # define the function to train a mini-batch.
        i = T.iscalar()  # define the index of mini-bathces
        train_mb = theano.function(
            [i],  # the input
            cost,  # the symbolic mathematical computation of cost
            updates=updates,
            givens={
                self.x:
                    training_x[i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y:
                    training_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # define the function to compute the accuracy of training.
        # mini-batches.
        train_mb_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    training_x[
                    i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y:
                    training_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # define the function to compute the accuracy of validation
        # mini-batches.
        validate_mb_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    validation_x[
                    i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y:
                    validation_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # define the function to compute the accuracy of test mini-batches.
        test_mb_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[
                    i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y:
                    test_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # define the prediction of test mini-batches.
        test_mb_pred = theano.function(
            [i],
            self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[
                    i * mini_batch_size: (i + 1) * mini_batch_size],
                # self.y:
                #     test_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # do the actual training.
        best_validation_accuracy = 0.0
        train_accuracys = []  # Save the accuracy of every epoch.
        validation_accuracys = []  # Save the accuracy of every epoch.
        print("epochs: {0} num_training_batches: {1}".format(epochs,
                                                             num_training_batches))
        for epoch in range(epochs):
            for mini_batch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + mini_batch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_train = train_mb(mini_batch_index)
                if (iteration + 1) % num_training_batches == 0:
                    # has train a epoch.
                    train_accuracy = \
                        np.mean([train_mb_accuracy(j) for j in range(
                            num_training_batches)])
                    validation_accuracy = \
                        np.mean([validate_mb_accuracy(j) for j in range(
                            num_validation_batches)])
                    print("Epoch {0}: train accuracy:{1:.2%} and validation "
                          "accuracy: {2:.2%}".format(
                        epoch, train_accuracy, validation_accuracy))
                    train_accuracys.append(train_accuracy)
                    validation_accuracys.append(validation_accuracy)
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = \
                                np.mean([test_mb_accuracy(j) for j in range(
                                    num_test_batches)])
                            print(
                                "The corresponding test accuracy is {"
                                "0:.2%}".format(
                                    test_accuracy))
        print("Finished training network.")
        print(
            "Best validation accuracy is {0:.2%} obtained at iteration {"
            "1}".format(
                best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy is {0:.2%}".format(test_accuracy))
        return train_accuracys, validation_accuracys, best_validation_accuracy


# The combination of convolution with max-pooling layer.
class ConvPoolLayer(object):

    def __init__(self, image_shape, filter_shape, pool_size=(2, 2),
                 activation_func=sigmoid):
        """The construct function of convolution and max-pooling layer.

        Args:
            self:
            image_shape: The inputs shape. It is a tuple of lengh 4, whose
             entries are mini batch size, the number of input feature
             maps, the filter height and the filter width.
            filter_shape: The convolution filter shape. It is a tuple of
            lengh 4, whose entries are the number of filters, the number of
            input feature maps, the filter height and the filter width.
            pool_size: The max-pooling size. It is a tuple of length 2,
            whose entries are x and y.
            activation_func: The activation function. Default is sigmoid.
        """
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.activation_func = activation_func
        # initialize the weights and biases.

        # calc the output length of conv and max-pooling layer.
        n_out = filter_shape[0] \
                * np.prod(np.asarray(image_shape[2:]) -
                          np.asarray(filter_shape[2:]) + 1) \
                / np.prod(pool_size)
        print("ConvPoolLayer n_out:%d" % n_out)
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out),
                                 size=filter_shape),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        print("ConvPoolLayer w.shape:")
        print(self.w.get_value().shape)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            name='b', borrow=True)
        print("ConvPoolLayer b.shape:")
        print(self.b.get_value().shape)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Construct the graph to compute layer output.

        Args:
            inpt: The input var.
            inpt_dropout: The dropouted input var.
            mini_batch_size: The mini batch size.
        """
        self.inpt = inpt.reshape(self.image_shape)
        conv_output = conv.conv2d(
            input=self.inpt,
            filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )
        pool_output = downsample.max_pool_2d(
            input=conv_output,
            ds=self.pool_size,
            ignore_border=True
        )
        self.output = self.activation_func(pool_output + self.b.dimshuffle(
            'x', 0, 'x', 'x'))
        self.output_dropout = self.output  # Don't dropout in conv layer.


# The full connected layer.
class FullConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_func=sigmoid, p_dropout=0.0):
        """The constructor function of full-connected layer.

        Args:
            n_in: The number of input neuron.
            n_out: The number of output neuron.
            activation_func: The activation function.
            p_dropout: The probability of dropout.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_func = activation_func
        self.p_dropout = p_dropout

        # Initialize the weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
            name='w', borrow=True)
        print("Full-connected layer w.shape:")
        print(self.w.get_value().shape)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        print("Full-connected layer b.shape:")
        print(self.b.get_value().shape)
        self.params = [self.w, self.b]


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Construct the graph to compute the layer output.

        Args:
            inpt: The input var.
            inpt_dropout: The dropouted input var.
            mini_batch_size: The mini batch size.
        """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_func(T.dot(self.inpt, self.w) + self.b)

        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size,
                                                         self.n_in)),
                                          p_dropout=self.p_dropout)
        self.output_dropout = self.activation_func(T.dot(self.inpt_dropout,
                                                         self.w) + self.b)

        # self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        # self.output = self.activation_func(T.dot(self.inpt, self.w) + self.b)
        # self.inpt_dropout = dropout_layer(
        #     inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        # self.output_dropout = self.activation_func(
        #     T.dot(self.inpt_dropout, self.w) + self.b)


# Define the softmax layer
class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        """The constructor function of softmax layer.

        Args:
            n_in: The number of input neuron.
            n_out: The number of output neuron.
            p_dropout: The probability of dropout.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize the weights and biases.
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        print("Softmax layer w.shape:")
        print(self.w.get_value().shape)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        print("Softmax layer b.shape:")
        print(self.b.get_value().shape)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Construct the graph to compute the softmax layer output.

        Args:
            inpt: The input var.
            inpt_dropout: The dropouted input var.
            mini_batch_size: The mini batch size.
        """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout,
                                            self.w) + self.b)

    def cost(self, net):
        """Return the log-likelihood cost.

        Args:
            net: The network

        Returns:
            The log-likelihood cost.
        """
        return -T.mean(
            T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        """Return the accuracy for the mini-batch

        Args:
            y: The desired output

        Returns:
            The accuracy for the mini-batch
        """
        return T.mean(T.eq(y, self.y_out))


def size(data):
    """Return the size of the dataset `data`.

    Args:
        data: The data.

    Returns:
        The size of dataset.
    """
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    """Return layer has been dropout.

    Args:
        layer: The layer.
        p_dropout: The probability of dropout.

    Returns:
        The layer has been dropout.
    """
    srs = shared_randomstreams.RandomStreams(np.random.RandomState(
        0).randint(999999))
    mask = srs.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX) / (1 - p_dropout)


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


if __name__ == "__main__":
    # x = T.dscalar('x')
    # y = T.dscalar('y')
    # z = x + y
    # f = function([x, y], z)
    # print f(2, 3)
    # print("openmp:" + str(theano.config.openmp))
    # print("device:" + str(theano.config.device))
    # print("floatX:" + str(theano.config.floatX))

    training_data, validation_data, test_data = load_mnist_data_shared()
    print("The shape of training_data:")
    print(training_data[0].get_value().shape)
    print("The shape of validation_data:")
    print(validation_data[0].get_value().shape)
    print("The shape of test_data:")
    print(test_data[0].get_value().shape)
    mini_batch_size = 10
    network = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                     filter_shape=(20, 1, 5, 5),
                                     activation_func=ReLu),
                       ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                     filter_shape=(40, 20, 5, 5),
                                     activation_func=ReLu),
                       FullConnectedLayer(n_in=(40 * 4 * 4), n_out=100,
                                          p_dropout=0.5, activation_func=ReLu),
                       SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)],
                      mini_batch_size=mini_batch_size)
    # network = Network([FullConnectedLayer(n_in=784, n_out=100,
    #                                       activation_func=ReLu, p_dropout=0.5),
    #                    SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)],
    #                   mini_batch_size=mini_batch_size)
    train_accuracys, validation_accuracys, best_test_accuracy = \
        network.SGD(
            training_data=training_data,
            epochs=60,
            mini_batch_size=mini_batch_size,
            eta=0.1,
            validation_data=validation_data,
            test_data=test_data,
            lmbda=5.0)
    plot_accuracy(train_accuracys, validation_accuracys, best_test_accuracy)
