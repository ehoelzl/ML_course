import numpy as np


def split_dataset(x, y, jet_col):
    """Splits the dataset into 3 categories (0 jets, 1 jet and more than 1 jet)
    :param x: Feature matrix (initial)
    :param y: Labels
    :param jet_col: Column of jet category (default=22)
    :return: x_0, y_0, x_1, y_1, x_2, y_2 The feature matrix and labels for each category
    """
    jets_0_inds, jets_1_inds, jets_rest_inds = np.where(x[:, jet_col] == 0), np.where(x[:, jet_col] == 1), np.where(x[:, jet_col] > 1)
    x_0, y_0 = x[jets_0_inds], y[jets_0_inds]
    x_1, y_1 = x[jets_1_inds], y[jets_1_inds]
    x_more, y_more = x[jets_rest_inds], y[jets_rest_inds]
    return x_0, y_0, x_1, y_1, x_more, y_more

def get_feature_columns(x):
    """Returns the indices of columns that have non null variance (i.e. columns where features vary)
    :param x: Feature matrix
    :return: np.array List of indices where the variance is > 0
    """
    std = np.std(x, axis=0)
    return np.where(std > 0)[0]

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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def standardize_matrix_columns(x):
    means = np.mean(x, axis=0)
    x = x - means
    std = np.std(x, axis=0)
    return x / std
    

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tiled = np.tile(np.vstack(x), degree-1)
    poly = np.power(tiled, np.arange(2, degree+1))
    return poly

def add_degrees(x, column, degree):
    if degree < 2:
        return x
    poly = build_poly(x[:, column], degree)
    return np.c_[x[:, :column+1], poly, x[:, column+1:]]