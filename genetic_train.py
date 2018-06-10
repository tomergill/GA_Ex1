import numpy as np
from chromosome import *
from sys import argv


def create_pool(sizes, pool_size, activ_func):
    pool = [Chromosome(sizes, activ_func) for _ in xrange(pool_size)]
    return pool


def neglogloss(y_hat, y):
    return -np.log(y_hat[range(y_hat.shape[0], y])


def accuracy_and_loss(net, images, labels, size=100):
    x = images[np.random.choice(images.shape[0], size), :]
    y = labels
    out = net.forward(x)
    loss = neglogloss(out, y).sum()
    acc = (np.argmax(out) == y).sum()
    return acc, loss


def train_on(pool, crossover, generations, train_images, train_labels):
    print "+------------+-----------+------------+-----------+-----------+----------+"
    print "| Generation | Gen. Time | Worst Loss | Worst Acc | Best Loss | Best Acc |"
    print "+------------+-----------+------------+-----------+-----------+----------+"
    for i in xrange(generations):
        pool_acc, pool_loss = zip(*map(lambda x: accuracy_and_loss(x, train_images, train_labels), pool))




def ranking(pool):
    pass


def main(sizes):
    # parameters
    activ_func = relu
    pool_size = 100
    generations = 100

    # load data
    dataloader = MNIST(return_type="numpy")
    train_images, train_labels = dataloader.load_training()
    test_images, test_labels = dataloader.load_testing()
    train_images = train_images.astype(np.float) / 255.0
    test_images = test_images.astype(np.float) / 255.0

    pool = create_pool(sizes, pool_size, activ_func)
    best = train_on(pool, None, generations)  # todo crossover


if __name__ == '__main__':
    if len(argv) == 1:
        sizes = [784, 200, 100, 40, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
