from chromosome import *
from sys import argv
from time import time
from mnist import MNIST


def mutate_by_row(c, prob):
    """
    Mutate each row in each weight in mutation probability (adding gaussian noise)
    :type c: Chromosome
    :param c: Chromosome to mutate
    :param prob: Mutation Probability
    :return: The mutated chromosome
    """
    mu, sigma = 0, 0.01
    new_layers = []
    for W, b in c.layers:
        new_W = W + np.outer(np.random.binomial(1, prob, size=W.shape[0]),
                             np.ones(W.shape[1])) * np.random.normal(mu, sigma, size=W.shape)
        new_b = b + np.random.binomial(1, prob, size=b.shape[0]) * np.random.normal(mu, sigma, size=b.shape[0])
        new_layers.append((new_W, new_b))
    return Chromosome(new_layers, c.activation_func, False)


def mutate_all_weight(c, prob):
    """
    Mutate each weight in mutation probability (adding gaussian noise)
    :param c: Chromosome to mutate
    :param prob: Mutation Probability
    :return: The mutated chromosome
    """
    mu, sigma = 0, 0.01
    new_layers = []
    for W, b in c.layers:
        new_W = W + np.random.binomial(1, prob) * np.random.normal(mu, sigma, size=W.shape)
        new_b = b + np.random.binomial(1, prob) * np.random.normal(mu, sigma, size=b.shape[0])
        new_layers.append((new_W, new_b))
    return Chromosome(new_layers, c.activation_func, False)


def cross_over_by_row(c1, c2):
    """
    Chooses each row (neuron) in each weight of the child from one of the parents randomly.
    Also chooses activation function from one of the parents randomly.
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1: Parent #1
    :param c2: Parent #2
    :return: Child chromosome
    """
    layers = []
    for (w1, b1), (w2, b2) in zip(c1.layers, c2.layers):
        v = np.random.randint(2, size=w1.shape[0])
        ones = np.ones(w1.shape[1])
        m = v.reshape(v.shape[0], 1).dot(ones.reshape(1, ones.shape[0]))
        w = w1 * m + w2 * (1 - m)
        b = v * b1 + (1 - v) * b2
        layers.append((w, b))
    return Chromosome(layers=layers, activ_func=c1.activation_func if np.random.randint(2) else c2.activation_func,
                      initialize=False)


def cross_over_by_layer(c1, c2):
    """
    Chooses each layer (weights and bias) of the child from one of the parents randomly.
    Also chooses activation function from one of the parents randomly.
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1: Parent #1
    :param c2: Parent #2
    :return: Child chromosome
    """
    v = np.random.randint(2, size=c1.n)
    layers = []
    for i in range(len(v)):
        layers.append(c1.layers[i] if v[i] else c2.layers[i])
    return Chromosome(layers, c1.activation_func if np.random.randint(2) else c2.activation_func, initialize=False)


def cross_over_by_element(c1, c2):
    """
    Chooses each element in each weight in each layer of the child from one of the parents randomly.
    Also chooses activation function from one of the parents randomly.
    :type c1: Chromosome
    :type c2: Chromosome
    :param c1: Parent #1
    :param c2: Parent #2
    :return: Child chromosome
    """
    layers = []
    for (w1, b1), (w2, b2) in zip(c1.layers, c2.layers):
        v = np.random.randint(2, size=w1.shape)
        w = w1 * v + w2 * (1 - v)
        v = np.random.randint(2, size=b1.shape[0])
        b = b1 * v + b2 * (1 - v)
        layers.append((w, b))
    return Chromosome(layers, c1.activation_func if np.random.randint(2) else c2.activation_func, initialize=False)


def create_pool(sizes, pool_size, activ_funcs):
    """
    Creates a starting population of chromosomes.
    :param sizes: A list of the sizes of the layers: [input_size, hidden_1, ..., output_size]
    :param pool_size: How many different chromosomes to create in population
    :param activ_funcs: A list of activation functions to choose from (each chromosome will have one)
    :return: A list with Chromosome objects
    """
    pool = [Chromosome(sizes, activ_funcs[np.random.choice(range(len(activ_funcs)), 1)[0]]) for _ in xrange(pool_size)]
    return pool


def random_set_from(images, labels, size=100):
    """
    Return a random set of different examples
    :param images: Images to choose from
    :param labels: Labels to choose from
    :param size: How many examples to choose
    :return: exampples' images, examples' labels
    """
    indices = np.random.choice(images.shape[0], size)
    x = images[indices, :]
    y = labels[indices]
    return x, y


def accuracy_and_loss(net, X, Y, return_preds=False):
    """
    Computes the outputs on inputs, then computes loss and accuracy.
    This is done in a  batched way.
    :param net: Neural Net (Chromosome)
    :param X: Images matrix
    :param Y: Labels vector
    :param return_preds: If True, also returns the predictions on X
    :return: accuracy, avearge loss [, predictions list]
    """
    # total_loss = good = 0.0
    # for x, y in zip(X, Y):
    #     out = net.forward(x)[0]
    #     total_loss += neglogloss(out, y)
    #     good += (np.argmax(out) == y)
    # return good / X.shape[0], total_loss / X.shape[0]
    batch_size = 256
    acc = loss = 0.0
    preds_list = []
    for i in range(0, X.shape[0], batch_size):
        x, y = X[i:i + batch_size], Y[i:i + batch_size]
        out = batched_forward(net, x)
        preds = np.argmax(out, axis=1)
        acc += (preds == y).sum()
        loss += batched_neglogloss(out, y).sum()
        if return_preds:
            preds_list.extend(preds)

    acc /= float(X.shape[0])
    loss /= float(X.shape[0])
    if return_preds:
        return acc, loss, preds_list
    return acc, loss


def batched_forward(ch, X):
    """
    Computes forward pass of chromosome for batched input
    :type ch: Chromosome
    :param ch: Net to compute on
    :param X: Input matrix (batched inputs) - each row is a single example
    :return: A probabilities matrix (sized examples_number*classes_number) - each row is output for a single example
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
    """
    Computes negative log loss for batch
    :param Y_hat: Probabilities matrix, row wise (each row's sum is 1, all non-negative)
    :param Y: Correct tags (indices) vector (each corrosponding to a row in Y_hat)
    :return: Loss vector (each element corrosponds to a row in Y_hat & an element from Y)
    """
    return -np.log(Y_hat[range(Y_hat.shape[0]), Y])


def mate(ranking, crossover, elitism, elitism_fraction):
    """
    Crosses over the chromosomes in the population, based on the ranking ("mating tickets").
    :param ranking: A list ordered in ranking order - best is index 0, worst is last.
    :param crossover: The crossover function to use
    :param elitism: If True, uses elitisim
    :param elitism_fraction:
        If elitism is on, then this fraction of the best chromosomes are copied as are to the new population, and the
        worst fraction are eliminated before mating.
    :return: The new population of children
    """
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


def rank(pool, images, labels, sort_by_loss=True, use_random_set=True):
    """
    Gives every chromosome a fitness value and then rank them by it.
    :param pool: The population (list of Chromosomes)
    :param images: Images matrix to rank by
    :param labels: Labels matrix to rank by
    :param sort_by_loss: If True the best is the chromosome with the lowest loss, otherwise highest accuracy
    :param use_random_set: If true uses a random set from images and labels
    :return: The population sorted by rank (best is index 0, worst is last)
    """
    if use_random_set:
        x, y = random_set_from(images, labels)
    else:
        x, y = images, labels
    results = map(lambda c: accuracy_and_loss(c, x, y), pool)
    pool_acc, pool_loss = zip(*results)

    # best, worst = np.argmax(pool_acc), np.argmin(pool_acc)
    # best_acc, worst_acc = pool_acc[best], pool_acc[worst]
    # best_loss, worst_loss = pool_loss[best], pool_loss[worst]

    zipped_pool = zip(pool, pool_loss) if sort_by_loss else zip(pool, pool_acc)
    ranking = sorted(zipped_pool, key=lambda x: x[1], reverse=not sort_by_loss)
    return zip(*ranking)[0]  # , worst_loss, worst_acc, best_loss, best_acc


def split_train_dev(images, labels, part=0.2, num_classes=10):
    """
    Split the images and labels to training set and validation set, making sure the validation set is balanced
        (classes-wise)
    :param images: matrix
    :param labels: vector
    :param part: The fraction of the validation set taken from images and labels (i.e 0.2 is 20% of set)
    :param num_classes: Number of classes
    :return: train_images, train_labels, dev_images, dev_labels
    """
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
             mutation_prob, mutate, elitism=False, elitism_fraction=0.05):
    """
    Trains using evolutionary algorithm.
    :param pool: Population (list of chromosomes)
    :param crossover: Crossover function to use (gets 2 Chromosomes and returns child Chromosome)
    :param generations: Number of generations to train
    :param train_images: Training set's images matrix
    :param train_labels: Training set's labels vector
    :param dev_images: Validation set's images matrix
    :param dev_labels: Validation set's labels vector
    :param mutation_prob: The probability to mutate
    :param mutate: Mutation function (gets a Chromosome and a mutation probability and returns a mutated Chromosome)
    :param elitism: If True, uses elitism with elitism_fraction
    :param elitism_fraction:
        If elitism is True, this is the fraction of best Chromosomes that are taken as are to the next generation's
        population, and this is the fraction of worst Chromosomes that are eliminated before getting to mate
    :return: The best Chromosome (on the validation set)
    """
    print "+------------+-----------+------------+-----------+-----------+----------+"
    print "| Generation | Gen. Time | Worst Loss | Worst Acc | Best Loss | Best Acc |"
    print "+------------+-----------+------------+-----------+-----------+----------+"
    start = time()
    for i in xrange(generations - 1):
        ranking = rank(pool, train_images, train_labels)
        new_pool = mate(ranking, crossover, elitism, elitism_fraction)
        pool = map(lambda c: mutate(c, mutation_prob), new_pool)

        if i % 100 == 99:
            dev_ranking = rank(ranking, dev_images, dev_labels, sort_by_loss=False, use_random_set=False)
            best, worst = dev_ranking[0], dev_ranking[-1]
            best_acc, best_loss = accuracy_and_loss(best, dev_images, dev_labels)
            worst_acc, worst_loss = accuracy_and_loss(worst, train_images, train_labels)
            print "| {:^10} | {:^8.4f}s | {:^10.4f} | {:^8.4f}% | {:^9.4f} | {:^7.4f}% |".format(
                i + 1, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
            start = time()
        if i % 1000 == 999:
            mutation_prob -= 0.005

    ranking = rank(pool, dev_images, dev_labels, sort_by_loss=False, use_random_set=False)
    best, worst = ranking[0], ranking[-1]
    best_acc, best_loss = accuracy_and_loss(best, dev_images, dev_labels)
    worst_acc, worst_loss = accuracy_and_loss(worst, train_images, train_labels)
    print "| {:^10} | {:^8.4f}s | {:^10.4f} | {:^8.4f}% | {:^9.4f} | {:^7.4f}% |".format(
        generations, time() - start, worst_loss, worst_acc * 100.0, best_loss, best_acc * 100.0)
    print "+------------+-----------+------------+-----------+-----------+----------+"
    return ranking[0]


def main(sizes):
    """
    Main function. Loads data, creates initial population, trains it and tests it on test. writes predictions to file.
    :param sizes: List of sized for the net, starting with input size and ending with output size
    :return: None
    """
    # parameters
    activ_funcs = [relu, sigmoid, tanh]
    pool_size = 150
    generations = 1000
    mutation_prob = 0.05
    mutate = mutate_by_row
    crossover = cross_over_by_layer
    elitism = True
    elitism_fraction = 0.1

    print "* Pool Size = {}".format(pool_size)
    print "* Generations = {}".format(generations)
    print "* Mutation Prob = {}".format(mutation_prob)
    print "* Crossover by {}".format(
        {cross_over_by_row: "Row", cross_over_by_element: "Element", cross_over_by_layer: "Layer"}[crossover])
    print "* Elitism is {}".format("ON" if elitism else "OFF")
    if elitism:
        print "\t# Elitism is {}%, which means {} chromosomes".format(elitism_fraction * 100.0,
                                                                      int(elitism_fraction * pool_size))
    print ""

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
                    mutate, elitism, elitism_fraction)
    print ""

    start = time()
    acc, loss, preds = accuracy_and_loss(best, test_images, test_labels, return_preds=True)
    print "Accuracy on Test Set is {}% and Loss is {}".format(acc * 100.0, loss)
    print "It took {} seconds".format(time() - start)
    with open("ga_test.pred", "w") as f:
        f.writelines(map(lambda x: str(int(x)) + "\n", preds))


if __name__ == '__main__':
    if len(argv) == 1:
        sizes = [784, 200, 100, 40, 10]
    else:
        sizes = [784] + map(int, argv[1:]) + [10]
    main(sizes)
