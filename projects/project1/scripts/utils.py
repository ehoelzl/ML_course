import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    """ Splits the dataset into train/test data
    
    :param x: Feature matrix
    :param y: Labels
    :param ratio: Ratio for train
    :param seed: Random seed
    :return: x_train, y_train, x_test, y_test
    """
    # set seed
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])  # Get random permutations of the indices
    num_train = int(ratio * x.shape[0])
    train_indices, test_indices = indices[:num_train], indices[num_train:]  # Split indices into train and test
    
    train_x, train_y = x[train_indices], y[train_indices]
    test_x, test_y = x[test_indices], y[test_indices]
    
    return train_x, train_y, test_x, test_y
