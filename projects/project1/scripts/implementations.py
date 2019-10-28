import numpy as np

from costs import compute_logistic_loss, compute_mse_loss, compute_reg_logistic_loss_l2
from gradients import compute_logistic_gradient, compute_mse_gradient
from proj1_helpers import batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma, _print=True):
    """
    Linear regression (Least squares) using gradient descent (using L1 Regularization)
    
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
        if n_iter % 100 == 0 and _print:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, _print=True):
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
        if n_iter % 1000 == 0 and _print:
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
    aI = (2 * N * lambda_) * np.identity(D)
    left = tx.T.dot(tx) + aI
    right = tx.T.dot(y)
    w = np.linalg.solve(left, right)
    loss = compute_mse_loss(y, tx, w) + (2 * lambda_ * np.linalg.norm(w) ** 2)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, _print=True):
    """
    Logistic regression using gradient descent of SGD
    
    :param y: Labels
    :param tx: Feature matrix
    :param initial_w: Initial weights
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w)
        loss = compute_logistic_loss(y, tx, w)
        
        # Update parameters
        w = w - gamma * dw
        # store w and loss
        if n_iter % 100 == 0 and _print:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, _print=True):
    """
    Regularized (using L2 regularization) Logistic regression using gradient descent or SGD
    
    :param y: Labels
    :param tx: Feature matrix
    :param lambda_: regularization parameter
    :param initial_w: Initial weights
    :param max_iters: Max iterations
    :param gamma: Step size
    :return: weights, loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w) + (lambda_ * w)
        loss = compute_reg_logistic_loss_l2(y, tx, w, lambda_)
        # Update parameters
        w = w - gamma * dw
        # store w and loss
        if n_iter % 100 == 0 and _print:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))
    loss = compute_reg_logistic_loss_l2(y, tx, w, lambda_)
    return w, loss

