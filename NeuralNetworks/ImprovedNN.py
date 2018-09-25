#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The improved multi-layer feed forward neural networks.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/14 14:28
"""
# Stand library
import json
import sys

# Third-party library
import numpy as np
import Utils
import matplotlib.pyplot as plt


### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Calc the quadratic cost associated with an output "a" and desired
         output "y"

        Args:
            a: output
            y: desired output

        Returns:
            The cost
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Calc the error delta from the output layer. The delta is used to calc
         weights delta.

        Args:
            z: the input of of neurons.
            a: the output of neurons.
            y: the desired output of neurons.

        Returns:
            The error delta used to calc weights delta.
        """
        return np.multiply((a - y), sigmoid_prime(z))


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Calc the cross entropy cost associated with an output "a" and desired
         output "y"

        Args:
            a: output
            y: desired output

        Returns:
            The cost
        """
        # Use nan_to_num to change nan to 0.
        # When y == a, log(0) is -inf.
        # -inf * 0 = nan.
        return np.sum(np.nan_to_num(np.multiply(-y, np.log(a))) -
                      np.nan_to_num(np.multiply((1 - y), np.log(1 - a))))

    @staticmethod
    def delta(z, a, y):
        """Calc the error delta from the output layer. The delta is used to calc
         weights delta.

        Args:
            z: the input of of neurons.
            a: the output of neurons.
            y: the desired output of neurons.

        Returns:
            The error delta used to calc weights delta.
        """
        # The entropy cost function can remove the derivative of sigmoid func.
        # So it can avoid the learning rate decline.
        return (a - y)

class LogLikelihoodCost(object):

    @staticmethod
    def fn(a, y):
        """Calc the soft max log-likelihood cost associated with an "a" and
        desired output "y"

        Args:
            a: output
            y: desired output

        Returns:
            The cost
        """
        return np.sum(-np.log(a))

    @staticmethod
    def delta(z, a, y):
        """Calc the error delta from the output layer. The delta is used to calc
         weights delta.

        Args:
            z: the input of of neurons.
            a: the output of neurons.
            y: the desired output of neurons.

        Returns:
            The error delta used to calc weights delta.
        """
        # The log-likelihood cost function can remove the derivative of softmax
        # func. So it can avoid the learning rate decline.
        return (a - y)


#### Network main class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, regularization='L2',
                 dropout=False, dropout_p=0.5):
        """The construct function of network class.

        Args:
            sizes: The number of neurons in the respective layers of the
            network. For example, ``[5, 3, 1]`` represent first layer has 5
            neurons and second layer has 3 neurons and third layer has 1 neuron.
            cost: The cost function. The default func is cross entropy cost
            func.
            regularization: The regularization method. Should be ``L1`` or
            ``L2`` or None. Default is ``L2``.
            dropout_p: The dropout probability. Default is ``0.5``.
        """
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.cost = cost
        if regularization != "L1" and regularization != "L2" and \
                regularization != None:
            raise ValueError("Regularization Should be ``L1`` or ``L2`` or "
                             "None.")
        self.regularization = regularization
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.default_weights_initialize()

    def default_weights_initialize(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of weights
        connecting to the same neuron. Initialize biases using a Gaussian
        distribution with mean 0 and stand deviation 1.

        """
        self.biases = [np.mat(np.random.randn(y, 1)) for y in self.sizes[1:]]
        self.weights = [np.mat(np.random.randn(y, x)) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weights_initialize(self):
        """Initialize weights and bias using a Gaussian distribution with
        mean 0 and stand deviation 1.

        """
        self.biases = [np.mat(np.random.randn(y, 1)) for y in self.sizes[1:]]
        self.weights = [np.mat(np.random.randn(y, x))
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Calc output of network with input a.

        Args:
            a: input

        Returns:
             output
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w * a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            eta_dynamic = False,
            eta_delta = 2,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.

        Args:
            training_data: The training data, like ``(x, y)``, with x
            representing input and y representing output.
            epochs: The train epochs.
            mini_batch_size: The batch size of every epoch.
            eta: The learning rate.
            eta_dynamic: Decide if eta is dynamic.
            eta_delta: The delta of eta change.
            lmbda: The regularization parameter ``lambda``
            evaluation_data: The validation or test data.
            monitor_evaluation_cost: If show evaluation cost in every epoch.
            monitor_evaluation_accuracy: If show evaluation accuracy in every
            epoch.
            monitor_training_cost: If show training cost in every epoch.
            monitor_training_accuracy: If show training accuracy in every epoch.

        Returns:
           evaluation_cost: The list containing evaluation cost in every epoch.
           evaluation_accuracy: The list containing evaluation accuracy in
           every epoch.
           training_cost: The list containing training cost in every epoch.
           training_accuracy: The list containing training accuracy in every
           epoch.

        """
        evaluation_cost = []
        evaluation_accuracy = []
        training_cost = []
        training_accuracy = []
        eta_change_num = 0

        for i in range(epochs):
            # Shuffle the training_data to get random mini_batches.
            np.random.shuffle(training_data)

            # Get all mini_batches from training_data.
            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in
                            range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                # Train the network weights and biases in every mini_batch.
                self.update_mini_batch(mini_batch, eta, lmbda,
                                       len(training_data))
            print("Epoch %d training complete." % i)
            # print("weights:")
            # print(self.weights)
            # print("biases:")
            # print(self.biases)
            if monitor_training_cost:
                cost = self.totalCost(training_data, lmbda, convert=False)
                training_cost.append(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True) / \
                           float(len(training_data))
                training_accuracy.append(accuracy)
            print("Training cost:%f   Training accuracy:%f" % (cost, accuracy))
            if monitor_evaluation_cost:
                cost = self.totalCost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=False) / \
                           float(len(evaluation_data))
                evaluation_accuracy.append(accuracy)
            print("Evaluation cost:%f   Evaluation accuracy:%f" % (
                cost, accuracy))

            # If eta is dynamic.
            # If accuracy is not promoted in 10 epochs, reduce the learning
            # rate eta. Eta will be reduced up to 10 times.
            if eta_dynamic and (i + 1) % 10 == 0:
                if evaluation_accuracy[i] <= evaluation_accuracy[i - 10]:
                    eta = eta / eta_delta
                    print("Now eta change to %f" % eta)
                    eta_change_num += 1
                    if eta_change_num == 10:
                        return training_cost, training_accuracy, \
                               evaluation_cost, evaluation_accuracy
        return training_cost, training_accuracy, evaluation_cost, \
               evaluation_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update weights and biases by applying gradient descent using
        propagation in a single mini batch.


        Args:
            mini_batch: A mini batch in a epoch.
            eta: Learning rate.
            lmbda: The regularization param lambda
            n: The total size of the training data set.

        """
        # Initialize the sum of biases and weights gradient for the cost
        # function C_x.
        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Use back propagation to get delta of nabla_b and nabla_w
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # print("delta_nabla_b:")
            # print(delta_nabla_b)
            # print("delta_nabla_w:")
            # print(delta_nabla_w)
            nabla_biases = [nb + dnb for nb, dnb in zip(nabla_biases,
                                                        delta_nabla_b)]
            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights,
                                                         delta_nabla_w)]

        if self.regularization == 'L2':
            self.weights = [
                (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch))
                * nw for w, nw in zip(self.weights, nabla_weights)]
        elif self.regularization == 'L1':
            self.weights = [
                (w - eta * (lmbda / n) * np.sign(w)) - (eta / len(mini_batch))
                * nw for w, nw in zip(self.weights, nabla_weights)]
        elif self.regularization == None:
            self.weights = [
                w - (eta / len(mini_batch))
                * nw for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_biases)]

    def backprop(self, x, y):
        """Return the tuple ``(nabla_b, nabla_w)`` representing the gradient
        for the cost function C_x.

        Args:
            x: input x
            y: output y

        Returns:
            The ``(nabla_b, nabla_w)`` are layer-by-layer lists of numpy
            arrays, similar to ``self.biases`` and ``self.weights``.

        """
        x = np.mat(x)
        y = np.mat(y)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Calculate the output by feed forward.
        activation = x  # save the activation of previous layer.
        if self.dropout:
            activation, random_tensor = dropout(activation, self.dropout_p)
        activations = [activation]  # save all activations of every layer.
        zs = []  # save all z of every layer.
        for b, w in zip(self.biases, self.weights):
            z = w * activation + b
            activation = sigmoid(z)
            if self.dropout:
                activation, random_tensor = dropout(activation, self.dropout_p)
            zs.append(z)
            activations.append(activation)

        # Calculate the gradient of the output layer by backward propagation.
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = delta * activations[-2].transpose()

        for l in range(2, self.num_layer):
            z = zs[-l]
            if self.cost == LogLikelihoodCost:
                sp = softmax_prime(z)
            else:
                sp = sigmoid_prime(z)
            delta = np.multiply(self.weights[-l + 1].transpose() * delta, sp)
            nabla_b[-l] = delta
            nabla_w[-l] = delta * activations[-l - 1].transpose()
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Calc the accuracy of validation data using current weights and
        biases. Calc the output using current weights and biases. Compare the
         output and desired output. If the output is the same as desired
         output. The accuracy count +1.

        Args:
            data: The validation data.
            convert: If the y is vectorized. The training_data is vectorized,
            but validation_data and test_data is not.

        Returns:
            The accuracy of validation data.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def totalCost(self, data, lmbda, convert=False):
        """Calc the total cost of validation data.

        Args:
            data: The validation data.
            lmbda: The regularization param lambda.
            convert: If y should be vectorized. The validation_data and
            test_data is not vectorized. So, they should be vectorized for calc
            cost.

        Returns:
            The total cost of validation data.
        """
        cost = 0.0
        for x, y in data:
            a = np.mat(x)
            a = self.feedforward(a)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)  # Add every example cost.
            break
        # Add regularization.
        if self.regularization == 'L2':
            cost += 0.5 * (lmbda / len(data)) * sum(
                np.linalg.norm(w) ** 2 for w in
                self.weights)
        elif self.regularization == 'L1':
            cost += 0.5 * (lmbda / len(data)) * sum(
                w.sum() for w in
                self.weights)
        elif self.regularization == None:
            cost += 0.0
        return cost


#### Loading a network from file
def load_network(file_name):
    """Load network from saved json file.

    Args:
        file_name: json file name.

    Returns:
        The loaded network.
    """
    f = open(file_name, "r")
    data = json.load(f)  # read network parameters from json data.
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["size"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """Vectorize 10 classified label values

    Args:
        j: class labels(0~9)

    Returns:
        Such as 5 -> 0000100000
    """
    vec = np.zeros((10, 1))
    vec[j] = 1.0
    return np.mat(vec)


def sigmoid(z):
    """sigmoid function

    Args:
        z: input

    Returns:
        sigmoid output
    """
    return 1.0 / (1 + np.exp(-z))


def softmax_prime(z):
    """The derivative function of softmax func.

    Args:
        z: input

    Returns:
        The output of sigmoid derivative func.
    """
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def sigmoid_prime(z):
    """The derivative function of sigmoid func.

    Args:
        z: input

    Returns:
        The output of sigmoid derivative func.
    """
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def dropout(x, level, random_tensor=None):
    """Dropout method.

    Args:
        x: input
        level: The dropout probability. For example 0.4 represent 4 zeros and 6
        staying the same in 10 input data.
        random_tensor: If not None, use the random_tensor to dropout.

    Returns:
        The out of dropout and random_tensor.
    """
    if level < 0. or level >= 1:  # The p should be in 0 ~ 1.0
        raise ValueError('Dropout level must be in interval (0, 1)')
    retain_prob = 1. - level

    # For example. The x has ten elements. The level is 4. Then random_tensor
    # may be [1, 1, 1, 1, 0, 1, 0, 1, 0, 0]. Input x will dropout 4 elements.
    if random_tensor == None:
        random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    x = np.multiply(x, random_tensor)
    x /= retain_prob  # rescale
    return x, random_tensor


def plot_accuracy(training_accuracy, evaluation_accuracy):
    """Plot the accuracy change of all epochs.

    Args:
        training_accuracy: The training accuracy list.
        evaluation_accuracy: The evaluation accuracy list.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, len(training_accuracy), 1)
    ax.plot(x, training_accuracy)
    ax.plot(x, evaluation_accuracy)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()


if __name__ == "__main__":
    # Test network using mnist directory data.
    my_net = Network([784, 100, 10], cost=LogLikelihoodCost,
                     regularization="L2")
    training_data, validation_data, test_data = Utils.load_mnist_data()
    print("Training data size:%d" % len(training_data))
    print("Validation data size:%d" % len(validation_data))
    print("Test data size:%d" % len(test_data))
    training_cost, training_accuracy, evaluation_cost, \
    evaluation_accuracy = my_net.SGD(training_data,
                                     epochs=100,
                                     mini_batch_size=10,
                                     eta=0.5,
                                     eta_dynamic=True,
                                     eta_delta=2,
                                     lmbda=5.0,
                                     evaluation_data=validation_data,
                                     monitor_training_cost=True,
                                     monitor_training_accuracy=True,
                                     monitor_evaluation_cost=True,
                                     monitor_evaluation_accuracy=True)
    print("training_accuracy:")
    print(training_accuracy)
    print("evaluation_accuracy:")
    print(evaluation_accuracy)
    plot_accuracy(training_accuracy, evaluation_accuracy)

    # Test network using Digits directory data.
    # my_net = Network([1024, 200, 10], regularization=None)
    # training_data = Utils.loadMatrixData("trainingDigits")
    # validation_data = Utils.loadMatrixData("testDigits", isTrainingData=False)
    # print("Training data size:%d" % len(training_data))
    # print("Validation data size:%d" % len(validation_data))
    # training_cost, training_accuracy, evaluation_cost, \
    # evaluation_accuracy = my_net.SGD(training_data,
    #                                  epochs=100,
    #                                  mini_batch_size=10,
    #                                  eta=0.5,
    #                                  eta_dynamic=True,
    #                                  eta_delta=2,
    #                                  lmbda=5.0,
    #                                  evaluation_data=validation_data,
    #                                  monitor_training_cost=True,
    #                                  monitor_training_accuracy=True,
    #                                  monitor_evaluation_cost=True,
    #                                  monitor_evaluation_accuracy=True)
    # print("training_accuracy:")
    # print(training_accuracy)
    # print("evaluation_accuracy:")
    # print(evaluation_accuracy)
    # plot_accuracy(training_accuracy, evaluation_accuracy)
