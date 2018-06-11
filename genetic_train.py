import numpy as np
from chromosome import *
from sys import argv





def mutate(c, prob):
    """
    :type c: Chromosome
    :param c:
    :param prob:
    :return:
    """
    mu, sigma = 0, 0.1
    for l in c.layers:
        l = (l[0] + np.random.binomial(1,prob) * np.random.normal(mu, sigma, size=l[0].shape), l[1] + np.random.binomial(1, prob) * np.random.normal(mu, sigma, size=l[1].shape[0]))


def cross_over_by_row(c1, c2):
    """
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1:
    :param c2:
    :return:
    """
    layers = []
    for (w1, b1), (w2, b2) in zip(c1.layers, c2.layers):
        v = np.random.randint(2, size=w1.shape[0])
        ones = np.ones(w1.shape[1])
        m = v.reshape(v.shape[0],1).dot(ones.reshape(1, ones.shape[0]))
        w = w1*m + w2 * (1-m)
        b = v*b1 + (1-v)*b2
        layers.append((w,b))
    return Chromosome(layers=layers, activ_func=c1.activation_func if v[0] == 0 else c2.activation_func, initialize=False)

def cross_over_by_layer(c1, c2):
    """
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1:
    :param c2:
    :return:
    """
    v = np.random.randint(2, size=c1.n)
    layers = []
    for i in range(len(v)):
        layers.append(c1.layers[i] if v[i] else c2.layers[i])
    return Chromosome(layers, c1.activation_func if v[0] == 0 else c2.activation_func, initialize=False)

def cross_over_by_element(c1,c2):
    """
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1:
    :param c2:
    :return:
    """
    layers = []
    for (w1, b1), (w2, b2) in zip(c1.layers, c2.layers):
        v = np.random.randint(2, size=w1.shape)
        w = w1 * v + w2 * (1-v)
        v = np.random.randint(2, size=b1.shape[0])
        b = b1*v + b2*(1-v)
        layers.append((w, b))
    return Chromosome(layers, c1.activation_func if v[0] == 0 else c2.activation_func, initialize=False)








def create_pool(sizes, pool_size, activ_func):
    pool = [Chromosome(sizes, activ_func) for _ in xrange(pool_size)]
    return pool


def neglogloss(y_hat, y):
    return -np.log(y_hat[range(y_hat.shape[0]), y])


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
    #cross_over_full_layer(1,2)
    if len(argv) == 1:
        sizes = [784, 200, 100, 40, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
