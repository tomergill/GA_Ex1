from chromosome import *
from sys import argv
from random import choice
from time import time
from mnist import MNIST





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
    return c


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


def random_set_from(images, labels, size=100):
    indices = np.random.choice(images.shape[0], size)
    x = images[indices, :]
    y = labels[indices]
    return x, y


def accuracy_and_loss(net, X, Y):
    total_loss = good = 0.0
    for x, y in zip(X, Y):
        out = net.forward(x)[0]
        total_loss += neglogloss(out, y)
        good += (np.argmax(out) == y)
    return good / X.shape[0], total_loss / X.shape[0]


def mate(ranking, crossover):
    new_pool = []
    tickets = {p: len(ranking) - i for i, p in enumerate(ranking)}

    while len(new_pool) < len(ranking):
        p1, p2 = choice(ranking), choice(ranking)
        if tickets[p1] <= 0 or tickets[p2] <= 0:
            continue
        tickets[p1] -= 1
        tickets[p2] -= 1
        new_pool.append(crossover(p1, p2))

    return new_pool


def rank(pool, train_images, train_labels, sort_by_loss=True):
    x, y = random_set_from(train_images, train_labels)
    results = map(lambda c: accuracy_and_loss(c, x, y), pool)
    pool_acc, pool_loss = zip(*results)

    best, worst = np.argmax(pool_acc), np.argmin(pool_acc)
    best_acc, worst_acc = pool_acc[best], pool_acc[worst]
    best_loss, worst_loss = pool_loss[best], pool_loss[worst]

    zipped_pool = zip(pool, pool_loss) if sort_by_loss else zip(pool, pool_acc)
    ranking = sorted(zipped_pool, key=lambda x: x[1], reverse=not sort_by_loss)
    return zip(*ranking)[0], worst_loss, worst_acc, best_loss, best_acc


def train_on(pool, crossover, generations, train_images, train_labels, mutation_prob):
    print "+------------+-----------+------------+-----------+-----------+----------+"
    print "| Generation | Gen. Time | Worst Loss | Worst Acc | Best Loss | Best Acc |"
    print "+------------+-----------+------------+-----------+-----------+----------+"
    start = time()
    for i in xrange(generations - 1):
        ranking, worst_loss, worst_acc, best_loss, best_acc = rank(pool, train_images, train_labels)
        new_pool = mate(ranking, crossover)
        pool = map(lambda c: mutate(c, mutation_prob), new_pool)

        if i % 100 == 99:
            print "| {:^10} | {:^8.f}s | {:^10.f} | {:^8.f}% | {:^9.f} | {:^7.f}% |".format(
                i+1, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
            start = time()

    ranking, worst_loss, worst_acc, best_loss, best_acc = rank(pool, train_images, train_labels)
    print "| {:^10} | {:^8.4f}s | {:^10.4f} | {:^8.4f}% | {:^9.4f} | {:^7.4f}% |".format(
        generations, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
    print "+------------+-----------+------------+-----------+-----------+----------+"
    return ranking[0]


def main(sizes):
    # parameters
    activ_func = relu
    pool_size = 100
    generations = 1000
    mutation_prob = 0.01
    crossover = cross_over_by_layer  # todo crossover

    # load data
    dataloader = MNIST(return_type="numpy")
    train_images, train_labels = dataloader.load_training()
    test_images, test_labels = dataloader.load_testing()
    train_images = train_images.astype(np.float) / 255.0
    test_images = test_images.astype(np.float) / 255.0

    pool = create_pool(sizes, pool_size, activ_func)
    best = train_on(pool, crossover, generations, train_images, train_labels, mutation_prob)
    acc, loss = accuracy_and_loss(best, test_images, test_labels)

    print "Accuracy on Test Set is {}% and Loss is {}".format(acc * 100.0, loss)


if __name__ == '__main__':
    #cross_over_full_layer(1,2)
    if len(argv) == 1:
        sizes = [784, 200, 100, 40, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
