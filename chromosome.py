import numpy as np


class Chromosome:
    """
    Class representing a chromosome, storing it's layers and it's activation function.
    """

    def __init__(self, layers, activ_func, initialize=True):
        """
        Constructor.
        :param layers:
            If initialize is true, then layers is a list of layer's dimensions, including the input size
            and output size. Otherwise, it's the layers themselves.
        :param activ_func: activation function (type func)
        :param initialize: If true initializes the layers using Glorot, otherwise receives the layers
        """
        if initialize:
            self.layers = []
            for s1, s2 in zip(layers[:-1], layers[1:]):
                W = initialize_weight(s1, s2)
                b = initialize_weight(s2)
                self.layers.append((W, b))
        else:
            self.layers = layers
        self.activation_func = activ_func
        self.n = len(self.layers)

    def forward(self, x):
        """
        Computes forward pass for chromosome
        :param x: input vector
        :return: output probabilities, a list containing the layers outputs before activation functions (zs)
        """
        zs = [x]
        g = self.activation_func
        h = x
        for W, b in self.layers[:-1]:
            z = W.dot(h) + b
            z = z.T
            zs.append(z)
            h = g(z)

        # last layer
        W, b = self.layers[-1]
        z = W.dot(h) + b
        z = z.T
        zs.append(z)
        return softmax(z), zs


def initialize_weight(s1, s2=None):
    """
    Glorot initialization.
    :param s1: Dimension #1
    :param s2: Dimension #2. If None, returns a vector.
    :return: Initialized matrix / vector
    """
    eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
    if s2 is None:
        return np.random.uniform(-eps, eps, s1)
    return np.random.uniform(-eps, eps, (s2, s1))


def softmax(x):
    """
    Computes the probabilities given a vector of scores.
    :param x: input vector
    :return: Probabilities vector (sum = 1.0)
    """
    exp = np.exp(x - x.max())
    return exp / exp.sum()


def sigmoid(x):
    """
    Logistic Sigmoid function.
    :param x: input
    :return:  sigmoid output
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLu activation function.
    :param x: input
    :return: relu output
    """
    return np.maximum(0, x)


tanh = np.tanh  # already implemented


def neglogloss(y_hat, y):
    """
    Computes the negative log loss.
    :param y_hat: Probabilities vector (sum = 1)
    :param y: The correct tag (index)
    :return: loss
    """
    return -np.log(y_hat[y])
