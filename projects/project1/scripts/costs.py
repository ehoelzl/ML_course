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


def compute_logistic_loss(y, tx, w):
    """Computes the loss of the logistic regression (- Log likelihood)
    
    :param y: Labels
    :param tx: Features
    :param w: Weight vector
    :return: Loss
    """
    xw = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(xw))) - y.T.dot(xw)
    return np.squeeze(loss)


def compute_rmse_loss(y, tx, w):
    return np.sqrt(2 * compute_mse_loss(y, tx, w))


def compute_reg_logistic_loss_l2(y, tx, w, lambda_):
    """ Computes the logistic loss regularized L2"""
    return compute_logistic_loss(y, tx, w) + ((1 / 2.0) * lambda_ * np.linalg.norm(w) ** 2)


def compute_reg_logistic_loss_l1(y, tx, w, lambda_):
    """ Computes the logistic loss regularized L1"""
    return compute_logistic_loss(y, tx, w) + (lambda_ * np.linalg.norm(w, 1))
