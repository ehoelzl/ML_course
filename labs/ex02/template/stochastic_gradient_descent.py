# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute the gradient for MSE of linear regression"""
    N = y.shape[0]
    e = y - (tx @ np.vstack(w)).flatten()
    return (-1/N) * (tx.T @ e)

def compute_subgradient(y, tx, w):
    """Computes the subgradient for MAE loss"""
    N = y.shape[0]
    e = y - (tx @ np.vstack(w)).flatten()
    e_sign = np.sign(e)
    return (-1/N) * tx.T @ e_sign


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter, data in enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters)):
        y_batch, tx_batch = data
        dw = compute_stoch_gradient(y_batch, tx_batch, w)
        loss = compute_mse_loss(y_batch, tx_batch, w)
        w = w - gamma * dw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def subgradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_subgradient(y, tx, w)
        loss = compute_mae_loss(y, tx, w)
        
        w = w - gamma * dw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws