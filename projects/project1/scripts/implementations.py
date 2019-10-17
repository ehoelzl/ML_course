from costs import compute_mse_loss, compute_mae_loss
from gradient_descent import compute_mse_gradient, compute_mae_subgradient
from utils import batch_iter
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression (Least squares) using gradient descent
    
    :param y: Labels
    :param tx: Feature Matrix
    :param initial_w: Initial weight vector
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_mse_gradient(y, tx, w)
        loss = compute_mse_loss(y, tx, w)

        # Update parameters
        w = w - gamma * dw
        # store w and loss
        if n_iter % 100 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression (Least squares) using stochastic gradient descent (batch size of 1)
    
    :param y: Labels
    :param tx: Feature Matrix
    :param initial_w: Initial weight vector
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
    w = initial_w
    batch_size = 1
    for n_iter, data in enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters)):
        # Fetch batch data
        y_batch, tx_batch = data
        dw = compute_mse_gradient(y_batch, tx_batch, w)
        loss = compute_mse_loss(y_batch, tx_batch, w)
        
        # Update weights
        w = w - gamma * dw
        if n_iter % 100 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    
    :param y: Labels
    :param tx: Feature matrix
    :return: weights, loss
    """
    right = tx.T.dot(y)
    left = tx.T.dot(tx)
    w = np.linalg.solve(left, right)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    :param y: Labels
    :param tx: Feature matrix
    :param lambda_: Regularization parameter
    :return: weights, loss
    """
    N, D = tx.shape
    aI = (2*N * lambda_) * np.identity(D)
    left = tx.T.dot(tx) + aI
    right = tx.T.dot(y)
    w = np.linalg.solve(left, right)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent of SGD
    
    :param y: Labels
    :param tx: Feature matrix
    :param initial_w: Initial weights
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
    
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized Logistic regression using gradient descent or SGD
    
    :param y: Labels
    :param tx: Feature matrix
    :param lambda_: regularization parameter
    :param initial_w: Initial weights
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
