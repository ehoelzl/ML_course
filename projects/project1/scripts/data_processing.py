import numpy as np
from utils import split_train_test


def split_dataset(x, y, jet_col, mass_col):
    """Splits the dataset into 6 categories (0 jets, 1 jet and more than 1 jet)
    :param x: Feature matrix (initial)
    :param y: Labels
    :param jet_col: Column of jet category (default=22)
    :return: x_0, y_0, x_1, y_1, x_2, y_2 The feature matrix and labels for each category
    """
    wm_0, nm_0 = np.where((x[:, jet_col] == 0) & (x[:, mass_col] > -999)), np.where(
            (x[:, jet_col] == 0) & (x[:, mass_col] == -999))
    wm_1, nm_1 = np.where((x[:, jet_col] == 1) & (x[:, mass_col] > -999)), np.where(
            (x[:, jet_col] == 1) & (x[:, mass_col] == -999))
    wm_2, nm_2 = np.where((x[:, jet_col] > 1) & (x[:, mass_col] > -999)), np.where(
            (x[:, jet_col] > 1) & (x[:, mass_col] == -999))
    
    x_0, y_0, x_0_nm, y_0_nm = x[wm_0], y[wm_0], x[nm_0], y[nm_0]
    x_1, y_1, x_1_nm, y_1_nm = x[wm_1], y[wm_1], x[nm_1], y[nm_1]
    x_2, y_2, x_2_nm, y_2_nm = x[wm_2], y[wm_2], x[nm_2], y[nm_2]
    assert (y_0.shape[0] + y_0_nm.shape[0] + y_1.shape[0] + y_1_nm.shape[0] + y_2.shape[0] + y_2_nm.shape[0]) == y.shape[0]
    return x_0, y_0, x_0_nm, y_0_nm, x_1, y_1, x_1_nm, y_1_nm, x_2, y_2, x_2_nm, y_2_nm


def normalize_columns(x):
    """Normalizes the columns of the matrix"""
    mins = np.min(x, axis=0)
    maxes = np.max(x, axis=0)
    return (x - mins) / (maxes - mins)


def standardize_columns(x):
    """Standardizes the columns of the matrix"""
    means = np.mean(x, axis=0)
    x = x - means
    std = np.std(x, axis=0)
    return x / std


def add_bias_column(tx):
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    return tx


def prepare_for_training(tx, y, cols, train_ratio, logistic=True, split=True):
    if logistic:
        tx = normalize_columns(tx)

    tx = standardize_columns(tx)
    tx = add_bias_column(tx)
    
    if cols is not None:
        tx = tx[:, cols]
    
    if split:
        x_train, y_train, x_test, y_test = split_train_test(tx, np.vstack(y), train_ratio)
    else:
        x_train, y_train, x_test, y_test = tx, np.vstack(y), np.array([]), np.array([])
    
    if logistic:
        y_train[np.where(y_train == -1)] = 0
    
    return x_train, y_train, x_test, y_test


def prepare_for_testing(tx, logistic=True):
    if logistic:
        tx = normalize_columns(tx)
    tx = standardize_columns(tx)
    tx = add_bias_column(tx)
    return tx
