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
import random
import sys

# Third-party library
import numpy as np
import Utils


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
        return a - y


#### Network main class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The construct function of network class.

        Args:
            sizes: The number of neurons in the respective layers of the
            network. For example, [5, 3, 1] represent first layer has 5 neurons
            and second layer has 3 neurons and third layer has 1 neuron.
            cost: The cost function. The default func is cross entropy cost
            func.
        """
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.cost = cost
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

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
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
        return training_cost, training_accuracy, evaluation_cost, \
               evaluation_data

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
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch))
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
        activations = [x]  # save all activations of every layer.
        zs = []  # save all z of every layer.
        for b, w in zip(self.biases, self.weights):
            z = w * activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Calculate the gradient of the output layer by backward propagation.
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = delta * activations[-2].transpose()

        for l in range(2, self.num_layer):
            z = zs[-l]
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
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)  # Add regularization.
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


def sigmoid_prime(z):
    """The derivative function of sigmoid func.

    Args:
        z: input

    Returns:
        The output of sigmoid derivative func.
    """
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


if __name__ == "__main__":
    my_net = Network([784, 100, 10])
    # x = np.array([[0, 1, 1, 1, 0, 1, 0, 1, 0, 1]]).transpose()
    # y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).transpose()
    # nabla_b, nabla_w = my_net.backprop(x, y)
    # print("nabla_b:")
    # print(nabla_b)
    # print("nabla_w:")
    # print(nabla_w)

    training_data, validation_data, test_data = Utils.load_mnist_data()
    print("Training data size:%d" % len(training_data))
    print("Validation data size:%d" % len(validation_data))
    print("Test data size:%d" % len(test_data))
    my_net.SGD(training_data,
               epochs=30,
               mini_batch_size=10,
               eta=0.5,
               lmbda=5.0,
               evaluation_data=validation_data,
               monitor_training_cost=True,
               monitor_training_accuracy=True,
               monitor_evaluation_cost=True,
               monitor_evaluation_accuracy=True)
