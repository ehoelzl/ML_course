import numpy as np


def split_dataset(x, y, jet_col):
    """Splits the dataset into 6 categories (0 jets, 1 jet and more than 1 jet)
    :param x: Feature matrix (initial)
    :param y: Labels
    :param jet_col: Column of jet category (default=22)
    :return: x_0, y_0, x_1, y_1, x_2, y_2 The feature matrix and labels for each category
    """
    
    ids_0 = np.where(x[:, jet_col] == 0)
    ids_1 = np.where(x[:, jet_col] == 1)
    ids_2 = np.where(x[:, jet_col] > 1)
    
    x_0, y_0 = x[ids_0], y[ids_0]
    x_1, y_1 = x[ids_1], y[ids_1]
    x_2, y_2 = x[ids_2], y[ids_2]
    
    assert (y_0.shape[0] + y_1.shape[0] + y_2.shape[0]) == y.shape[0]
    return x_0, y_0, x_1, y_1, x_2, y_2,


def normalize_columns(x):
    """Normalizes the columns of the matrix"""
    mins = np.nanmin(x, axis=0)
    maxes = np.nanmax(x, axis=0)
    return (x - mins) / (maxes - mins)


def standardize_columns(x):
    """Standardizes the columns of the matrix"""
    means = np.nanmean(x, axis=0)
    x = x - means
    std = np.nanstd(x, axis=0)
    return x / std


def add_bias_column(tx):
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    return tx


def add_mass_column(tx):
    vec = np.zeros((tx.shape[0], 1))
    vec[np.where(np.isnan(tx[:, 0]))] = 1
    return np.c_[tx, vec]


def prepare_for_training(tx, y, logistic=True):
    if logistic:
        tx = normalize_columns(tx)
        y[np.where(y == -1)] = 0
    
    tx = standardize_columns(tx)
    tx = add_mass_column(tx)
    tx = add_bias_column(tx)
    
    tx[np.isnan(tx)] = 0
    return tx, np.vstack(y)


def prepare_for_testing(tx, y, logistic=True):
    if logistic:
        tx = normalize_columns(tx)
    
    tx = standardize_columns(tx)
    tx = add_mass_column(tx)
    tx = add_bias_column(tx)
    tx[np.isnan(tx)] = 0
    
    if y is not None:
        return tx, np.vstack(y)
    
    return tx, None
