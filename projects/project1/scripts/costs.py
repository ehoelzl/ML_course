# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


def compute_mse_loss(y, tx, w):
    """ Computes Mean Squared Error of given input
    
    :param y: Labels (n, 1)
    :param tx: Feature matrix (n, d)
    :param w: Weight vector (d, 1)
    :return: MSE Loss
    """
    e = y - tx.dot(w)
    return np.mean(e ** 2) / 2.0


def compute_mae_loss(y, tx, w):
    """ Computes Mean Absolute Error
    
    :param y: Labels (n, 1)
    :param tx: Feature matrix (n, d)
    :param w: Weight vector (d, 1
    :return: MAE Loss
    """
    e = np.abs(y - tx.dot(w))
    return np.mean(e)


def compute_rmse_loss(y, tx, w):
    return np.sqrt(2 * compute_mse_loss(y, tx, w))
