import numpy as np
from utils import sigmoid


def compute_mse_gradient(y: np.array, tx: np.array, w: np.array):
    """
    Computes and return the gradient of the MSE loss function
    :param y: Labels (n, 1)
    :param tx : Feature matrix (n, d)
    :param w: Weight vector (d, 1)
    :return: MSE gradient
    """
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / len(e)
    return gradient


def compute_mae_subgradient(y: np.array, tx: np.array, w: np.array):
    """
    Computes and returns the subgradient of the MAE loss function
    :param y: Labels (n,1)
    :param tx: Feature Matrix (n, d)
    :param w: Weight vector (d, 1)
    :return: MAE Subgradient
    """
    sign_e = np.sign(y - tx.dot(w))
    subgradient = - tx.T.dot(sign_e) / len(sign_e)
    return subgradient


def compute_logistic_gradient(y: np.array, tx: np.array, w: np.array):
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient