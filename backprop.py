from chromosome import *
import numpy as np
from time import time
from sys import argv
from itertools import izip
from mnist import MNIST

# maps each activation function to it's derivative function
g2dg = {sigmoid: lambda x: sigmoid(x) * (1 - sigmoid(x)),
        relu: lambda x: (x > 0).astype(np.int),
        tanh: lambda x: 1 - np.square(tanh(x))}


def backprop(ch, x, y):
    """
    Computes the back propagation algorithm for a given chromosome
    :param y: correct tag
    :param x: input vector
    :param ch: Chromosome to backprop
    :type ch: Chromosome
    :return: loss and gradients (list of (gW, gb)s)
    """
    y_hat, zs = ch.forward(x)
    pred = np.argmax(y_hat)
    loss = neglogloss(y_hat, y)

    grads = []
    g = ch.activation_func
    dg = g2dg[g]

    # last layer
    W, b = ch.layers[-1]
    dL_dzn = y_hat.copy()
    dL_dzn[y] -= 1
    dL_dbn = dL_dzn
    dL_dWn = np.outer(dL_dzn, g(zs[-2]))
    grads.append((dL_dWn, dL_dbn))

    # rest of layers
    prev_W = W
    dL_dzi = dL_dzn
    most_layers = ch.layers[:-1]
    for i, (W, b) in enumerate(most_layers[::-1]):
        i = ch.n - 1 - i
        z = zs[i]
        dL_dzi = dL_dzi.dot(prev_W) * dg(z)
        gb = dL_dzi
        gW = np.outer(dL_dzi, g(zs[i - 1]))
        grads.append((gW, gb))
        prev_W = W

    return pred, loss, reversed(grads)


def sgd_update(ch, grads, lr):
    """
    Updates chromosome's weights using sgd update: w_new = w_old + lr*(d Loss / d w_old)
    :param ch: Chromosome to update
    :param grads: List of gradients, in the same order of the chromosome's layers
    :param lr: Learning rate
    :return: None
    """
    for layer, layer_grads in izip(ch.layers, grads):
        for weight, grad in izip(layer, layer_grads):
            weight -= lr * grad


def accuracy_and_loss_on(ch, images, labels, return_pred=False):
    """
    Computes the output on images, then computes accuracy and loss
    :param ch: Chromosome to test
    :param images: Images matrix
    :param labels: Labels vector
    :param return_pred: If true, returns the prediction (output of chromosome)
    :return: accuracy, average loss [, predictions list]
    """
    if return_pred:
        preds = []
    total_loss = good = 0.0
    for x, y in izip(images, labels):
        y_hat, _ = ch.forward(x)
        pred = np.argmax(y_hat)
        good += pred == int(y)
        total_loss += neglogloss(y_hat, y)
        if return_pred:
            preds.append(pred)
    if return_pred:
        return good / len(images), total_loss / len(images), preds
    return good / len(images), total_loss / len(images)


def train_on(ch, train_images, train_labels, dev_images, dev_labels, epochs=15, lr=0.001):
    """
    Trains the network, after each epoch test on validation. Prints in a pretty table.
    :param ch: Neural Network
    :param train_images: Train images matrix
    :param train_labels: Train images vecotr
    :param dev_images: Validation images matrix
    :param dev_labels: Validation images vector
    :param epochs: Number of epochs to train on
    :param lr: Learning rate
    :return: None
    """
    print "+-------+------------+-----------+----------+---------+------------+"
    print "| epoch | train loss | train acc | dev loss | dev acc | epoch time |"
    print "+-------+------------+-----------+----------+----------------------+"

    for i in xrange(epochs):
        start = time()
        total_loss = good = 0.0
        for x, y in izip(train_images, train_labels):
            pred, loss, grads = backprop(ch, x, y)
            good += pred == y
            total_loss += loss
            sgd_update(ch, grads, lr)
        train_acc, train_loss = good / len(train_images), total_loss / len(train_images)
        dev_acc, dev_loss = accuracy_and_loss_on(ch, dev_images, dev_labels)

        print "| {:^5} | {:010f} | {:8.4f}% | {:7f} | {:6.3f}% | {:08f}s |".format(
            i, train_loss, train_acc * 100.0, dev_loss, dev_acc * 100.0, time() - start)
    print "+-------+------------+-----------+----------+---------+------------+\n"


def main(sizes):
    """
    Main function. Loads data, creates net, trains it and tests it on test. writes predictions to file.
    :param sizes: List of sized for the net, starting with input size and ending with output size
    :return: None
    """
    lr = 0.001
    epochs = 30

    # load data
    dataloader = MNIST(return_type="numpy")
    train_images, train_labels = dataloader.load_training()
    test_images, test_labels = dataloader.load_testing()

    dev_indices = np.random.choice(len(train_images), int(0.2 * len(train_images)))
    train_indices = list(set(range(len(train_images))) - set(dev_indices))
    dev_images, dev_labels = train_images[dev_indices], train_labels[dev_indices]
    train_images, train_labels = train_images[train_indices], train_labels[train_indices]

    train_images = train_images.astype(np.float) / 255.0
    dev_images = dev_images.astype(np.float) / 255.0
    test_images = test_images.astype(np.float) / 255.0

    # create net, train and test
    net = Chromosome(sizes, relu)
    train_on(net, train_images, train_labels, dev_images, dev_labels, lr=lr, epochs=epochs)
    test_acc, test_loss, preds = accuracy_and_loss_on(net, test_images, test_labels, True)
    print "\nTest accuracy {}%, with loss of {}".format(test_acc * 100, test_loss)

    with open("bp_test.pred", "w") as f:
        f.writelines(map(lambda x: str(int(x)) + "\n", preds))


if __name__ == '__main__':
    if len(argv) == 1:
        sizes = [784, 200, 100, 40, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
