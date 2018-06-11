import numpy as np


class Chromosome:
    def __init__(self, layers, activ_func, initialize=True):
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
        zs = [x]
        g = self.activation_func
        h = x
        for W, b in self.layers[:-1]:
            z = W.dot(h) + b
            zs.append(z)
            h = g(z)

        # last layer
        W, b = self.layers[-1]
        z = W.dot(h) + b
        zs.append(z)
        return softmax(z), zs


def initialize_weight(s1, s2=None):
    eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
    if s2 is None:
        return np.random.uniform(-eps, eps, s1)
    return np.random.uniform(-eps, eps, (s2, s1))


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


tanh = np.tanh

