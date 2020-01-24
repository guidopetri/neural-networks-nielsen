#!/usr/bin/env python3

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, activation):
        for bias, weight in zip(self.biases, self.weights):
            activation = sigmoid(weight @ activation + bias)
        return activation

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the net using mini-batch SGD. If `test_data` is provided,
        partial progress per epoch will be shown.
        """

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for epoch in range(epochs):
            # shuffle training data in place
            np.random.shuffle(training_data)

            # create mini batches
            mini_batches = [training_data[x: x + mini_batch_size]
                            for x in range(0, n, mini_batch_size)]

            for i, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(mini_batch, eta)
                if i % 1000 == 0 and not test_data:
                    print("Epoch {}: mini-batch {} complete".format(epoch, i),
                          flush=True)

            if test_data:
                print("Epoch {}: {} / {}".format(epoch,
                                                 self.evaluate(test_data),
                                                 n_test),
                      flush=True)
            else:
                print("Epoch {} complete".format(epoch),
                      flush=True)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(x == y for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):

        # write matrix version
        raise NotImplementedError
        grad_b = [np.zeros(bias.shape) for bias in self.biases]
        grad_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [prior + delta
                      for prior, delta in zip(grad_b, delta_grad_b)]
            grad_w = [prior + delta
                      for prior, delta in zip(grad_w, delta_grad_w)]

        self.biases = [bias - (eta / len(mini_batch)) * posterior
                       for bias, posterior in zip(self.biases, grad_b)]
        self.weights = [weight - (eta / len(mini_batch)) * posterior
                        for weight, posterior in zip(self.weights, grad_w)]

    def backprop(self, x, y):
        grad_b = [np.zeros(bias.shape) for bias in self.biases]
        grad_w = [np.zeros(weight.shape) for weight in self.weights]

        # forward
        activation = x
        activations = [x]
        zs = []

        for bias, weight in zip(self.biases, self.weights):
            z = weight @ activation + bias
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # backward

        delta = (self.cost_derivative(activations[-1], y)
                 * sigmoid_derivative(zs[-1]))
        grad_b[-1] = delta
        grad_w[-1] = delta @ activations[-2].T

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sig = sigmoid_derivative(z)
            delta = self.weights[-layer + 1].T @ delta * sig

            grad_b[-layer] = delta
            grad_w[-layer] = delta @ activations[-layer - 1].T

        return (grad_b, grad_w)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    x = sigmoid(z)
    return x * (1 - x)


if __name__ == '__main__':
    # load the data in an actually proper format for a book like this
    import pickle

    with open('mnist_fixed.pckl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # unpack
        training_data, validation_data, test_data = data

    # init network
    net = Network([784, 10])
    net.SGD(training_data,
            epochs=30,
            mini_batch_size=10,
            eta=3,
            test_data=test_data)
