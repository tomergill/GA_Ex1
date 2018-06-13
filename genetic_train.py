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
    new_layers = []
    for W, b in c.layers:
        new_W = W + np.random.binomial(1,prob) * np.random.normal(mu, sigma, size=W.shape)
        new_b = b + np.random.binomial(1, prob) * np.random.normal(mu, sigma, size=b.shape[0])
        new_layers.append((new_W, new_b))
    return Chromosome(new_layers, c.activation_func, False)


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








def create_pool(sizes, pool_size, activ_funcs):
    pool = [Chromosome(sizes, activ_funcs[np.random.choice(range(len(activ_funcs)), 1)]) for _ in xrange(pool_size)]
    return pool


def random_set_from(images, labels, size=100):
    indices = np.random.choice(images.shape[0], size)
    x = images[indices, :]
    y = labels[indices]
    return x, y


def accuracy_and_loss(net, X, Y, return_preds=False):
    # total_loss = good = 0.0
    # for x, y in zip(X, Y):
    #     out = net.forward(x)[0]
    #     total_loss += neglogloss(out, y)
    #     good += (np.argmax(out) == y)
    # return good / X.shape[0], total_loss / X.shape[0]
    out = batched_forward(net, X)
    preds = np.argmax(out, axis=1)
    acc = (preds == Y).sum() / float(X.shape[0])
    loss = batched_neglogloss(out, Y).sum() / float(X.shape[0])

    if return_preds:
        return acc, loss, preds
    return acc, loss


def batched_forward(ch, X):
    """
    :type ch: Chromosome
    :param ch:
    :param X:
    :return:
    """
    H = X.T
    ones = np.ones(X.shape[0])
    for W, b in ch.layers[:-1]:
        Z = W.dot(H) + np.outer(b, ones)
        H = ch.activation_func(Z)
    W, b = ch.layers[-1]
    Z = W.dot(H) + np.outer(b, ones)

    # softmax
    Z = Z.T
    ones = np.ones(Z.shape[1])
    exps = np.exp(Z - np.outer(Z.max(axis=1), ones))
    preds = exps / np.outer(exps.sum(axis=1), ones)
    return preds


def batched_neglogloss(Y_hat, Y):
    return -np.log(Y_hat[range(Y_hat.shape[0]), Y])


def mate(ranking, crossover, elitism, elitism_fraction):
    new_pool = []
    size = len(ranking)

    if elitism:
        elitism_size = int(len(ranking) * elitism_fraction)
        new_pool.extend(ranking[:elitism_size])
        ranking = ranking[:-elitism_size]

    tickets = {p: len(ranking) - i for i, p in enumerate(ranking)}

    while len(new_pool) < size:
        p1, p2 = np.random.choice(ranking, 2)
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

    # best, worst = np.argmax(pool_acc), np.argmin(pool_acc)
    # best_acc, worst_acc = pool_acc[best], pool_acc[worst]
    # best_loss, worst_loss = pool_loss[best], pool_loss[worst]

    zipped_pool = zip(pool, pool_loss) if sort_by_loss else zip(pool, pool_acc)
    ranking = sorted(zipped_pool, key=lambda x: x[1], reverse=not sort_by_loss)
    return zip(*ranking)[0]  #, worst_loss, worst_acc, best_loss, best_acc


def split_train_dev(images, labels, part=0.2, num_classes=10):
    indices = range(images.shape[0])
    np.random.shuffle(indices)
    images, labels = images[indices], labels[indices]
    dev_indices = []
    examples_per_class = int(images.shape[0] * part / num_classes)
    counters = [0] * num_classes
    for i in xrange(images.shape[0]):
        label = int(labels[i])
        if counters[label] < examples_per_class:
            dev_indices.append(i)
            counters[label] += 1
        elif sum(counters) > (examples_per_class * num_classes):
            break
    train_indices = list(set(range(len(images))) - set(dev_indices))
    dev_images, dev_labels = images[dev_indices], labels[dev_indices]
    train_images, train_labels = images[train_indices], labels[train_indices]
    return train_images, train_labels, dev_images, dev_labels


def train_on(pool, crossover, generations, train_images, train_labels, dev_images, dev_labels,
             mutation_prob, elitism=False, elitism_fraction=0.05):
    print "+------------+-----------+------------+-----------+-----------+----------+"
    print "| Generation | Gen. Time | Worst Loss | Worst Acc | Best Loss | Best Acc |"
    print "+------------+-----------+------------+-----------+-----------+----------+"
    start = time()
    for i in xrange(generations - 1):
        ranking = rank(pool, train_images, train_labels)
        new_pool = mate(ranking, crossover, elitism, elitism_fraction)
        pool = map(lambda c: mutate(c, mutation_prob), new_pool)

        if i % 100 == 99:
            best, worst = ranking[0], ranking[-1]
            best_acc, best_loss = accuracy_and_loss(best, dev_images, dev_labels)
            worst_acc, worst_loss = accuracy_and_loss(worst, train_images, train_labels)
            print "| {:^10} | {:^8.4f}s | {:^10.4f} | {:^8.4f}% | {:^9.4f} | {:^7.4f}% |".format(
                i+1, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
            start = time()

    ranking = rank(pool, dev_images, dev_labels)
    best, worst = ranking[0], ranking[-1]
    best_acc, best_loss = accuracy_and_loss(best, dev_images, dev_labels)
    worst_acc, worst_loss = accuracy_and_loss(worst, train_images, train_labels)
    print "| {:^10} | {:^8.4f}s | {:^10.4f} | {:^8.4f}% | {:^9.4f} | {:^7.4f}% |".format(
        generations, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
    print "+------------+-----------+------------+-----------+-----------+----------+"
    return ranking[0]


def main(sizes):
    # parameters
    activ_funcs = [relu, sigmoid, tanh]
    pool_size = 100
    generations = 100
    mutation_prob = 0.001
    crossover = cross_over_by_row
    elitism = True
    elitism_fraction = 0.1

    # load data
    dataloader = MNIST(return_type="numpy")
    train_images, train_labels = dataloader.load_training()
    test_images, test_labels = dataloader.load_testing()
    train_images, train_labels, dev_images, dev_labels = split_train_dev(train_images, train_labels)
    train_images = train_images.astype(np.float) / 255.0
    dev_images = dev_images.astype(np.float) / 255.0
    test_images = test_images.astype(np.float) / 255.0

    pool = create_pool(sizes, pool_size, activ_funcs)
    best = train_on(pool, crossover, generations, train_images, train_labels, dev_images, dev_labels, mutation_prob,
                    elitism, elitism_fraction)
    acc, loss, preds = accuracy_and_loss(best, test_images, test_labels, return_preds=True)

    print "Accuracy on Test Set is {}% and Loss is {}".format(acc * 100.0, loss)
    with open("ga_test.pred", "w") as f:
        f.writelines(map(lambda x: str(int(x)) + "\n", preds))


if __name__ == '__main__':
    #cross_over_full_layer(1,2)
    if len(argv) == 1:
        sizes = [784, 200, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
