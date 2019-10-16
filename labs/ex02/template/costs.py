# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_mse_loss(y, tx, w):
    """Calculates the MSE loss"""
    N = y.shape[0]
    e = y - (tx @ np.vstack(w)).flatten()
    return np.linalg.norm(e)**2 / (2*N)

def compute_mae_loss(y, tx, w):
    N = y.shape[0]
    e = np.abs(y - (tx @ np.vstack(w)).flatten())
    return np.sum(e) / N