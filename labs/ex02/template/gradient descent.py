# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient for MSE of linear regression"""
    N = y.shape[0]
    e = y - (tx @ np.vstack(w)).flatten()
    return (-1/N) * (tx.T @ e)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        dw = compute_gradient(y, tx, w)
        loss = compute_mse_loss(y, tx, w)

        w = w - gamma * dw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws