import numpy as np
from implementations import least_squares_GD, ridge_regression
from costs import compute_mse_loss
from proj1_helpers import compute_accuracy

def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ridge(y, x, k_indices, k, lambda_):
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    w, loss_tr = ridge_regression(y_train, x_train, lambda_)
    
    loss_val = compute_mse_loss(y_val, x_val, w) + (2*lambda_*np.linalg.norm(w)**2)
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc


def cross_validation_ls_gd(y, x, k_indices, k, iters, gamma, lambda_):
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    initial_w = np.ones((x_train.shape[1], 1))
    w, loss_tr = least_squares_GD(y_train, x_train, initial_w, iters, gamma, lambda_=lambda_, _print=False)
    
    loss_val = compute_mse_loss(y_val, x_val, w) + (lambda_ * np.sum(np.abs(w)))
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc
    
"""
def cross_validation_ls_gd(y, x, k_fold, gammas, max_iters):
    loss_tr, loss_te = [], [] # Define loss accumulators for both
    acc_tr, acc_te = [], [] # Define accuracy accumulators
    k_indices = build_k_indices(y, k_fold) # TODO rename test to validation
    
    best_weights = None
    max_acc = 0
    for gamma in gammas:
        loss_tr_tmp, loss_te_tmp = [], [] 
        acc_tr_tmp, acc_te_tmp = [], [] 
        for k in range(k_fold):
            indices_te, indices_tr = k_indices[k], k_indices[~(np.arange(k_fold) == k)].reshape(-1)
            x_test, y_test = x[indices_te, :], y[indices_te]
            x_train, y_train = x[indices_tr, :], y[indices_tr]
            
            initial_w = np.ones((x_train.shape[1], 1))
            w, loss_train = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma, _print=False)
            
            loss_test = compute_mse_loss(y_test, x_test, w)
            acc_test, acc_train = compute_accuracy(y_test, x_test, w), compute_accuracy(y_train, x_train, w)
            
            print(f"Gamma={gamma}, loss_te={loss_test}, acc_te={acc_test}")
            if acc_test > max_acc:
                best_weights = w
                max_acc = acc_test
                
            loss_tr_tmp.append(loss_train)
            loss_te_tmp.append(loss_test)
            acc_tr_tmp.append(acc_train)
            acc_te_tmp.append(acc_test)
            
        loss_tr.append(loss_tr_tmp)
        loss_te.append(loss_te_tmp)
        acc_tr.append(acc_tr_tmp)
        acc_te.append(acc_te_tmp)
    return best_weights, loss_te, acc_te, loss_tr, acc_tr
"""