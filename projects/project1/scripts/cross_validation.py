import numpy as np
from implementations import reg_logistic_regression
from costs import compute_reg_logistic_loss
from proj1_helpers import compute_accuracy

from feature_expansion import enhance_features
from data_processing import prepare_for_training
import matplotlib.pyplot as plt 

def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation_reg_logistic(y, x, k_indices, k, iters, gamma, lambda_):
    """Performs one iteration of the k-fold cross validation using Least Square Gradient descent (and l1  regularization if lambda_ > 0)"""
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    initial_w = np.ones((x_train.shape[1], 1))
    w, loss_tr = reg_logistic_regression(y_train, x_train, lambda_, initial_w, iters, gamma, _print=False)
    
    loss_val = compute_reg_logistic_loss(y_val, x_val, w, lambda_)
    y_val[np.where(y_val == 0)] = -1
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc


def cross_validate_lambdas(y, x, k_fold, iters, gamma, lambdas):
    k_indices = build_k_indices(y, k_fold)
    accuracies, losses = [], []
    for lambda_ in lambdas:
        accuracy, losses_val = [], []
        
        for k in range(k_fold):
            w, loss_tr, loss_val, acc_val = cross_validation_reg_logistic(y, x, k_indices, k, iters, gamma, lambda_)
            #print(acc_val, loss_val)
            accuracy.append(acc_val)
            losses_val.append(loss_val)
        accuracies.append(accuracy)
        losses.append(losses_val)
    return accuracies, losses


def cross_validate_degrees(x, y, headers, gamma, lambda_, degrees, k_fold, title):
    max_iters = 1000
    k_indices = build_k_indices(y, k_fold)
    accuracies, losses = [], []
    
    for deg in degrees:
        x_new, proj_mat = enhance_features(x, headers, deg, plot=False, print_=False)
        x_train, y_train, _, _ = prepare_for_training(x_new, y, 0, split=False)
        accs, losses_val = [], []
        for k in range(k_fold):
            w, loss_tr, loss_val, acc = cross_validation_reg_logistic(y_train, x_train, k_indices, k, max_iters, gamma, lambda_)
            accs.append(acc)
            losses_val.append(loss_val)
        #print(f"Average accuracy for degrees up to {deg}={np.mean(accs)}")
        accuracies.append(accs)
        losses.append(losses_val)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    
    ax[0].plot(np.mean(accuracies, axis=1))
    ax[0].set_xticks(np.arange(len(degrees)));
    ax[0].set_xticklabels(degrees);
    ax[0].set_title("Mean validation accuracy")
    ax[0].set_xlabel("Polynomial expansion degree")
    
    ax[1].plot(np.mean(losses, axis=1))
    ax[1].set_xticks(np.arange(len(degrees)));
    ax[1].set_xticklabels(degrees);
    ax[1].set_title("Mean validation loss")
    ax[1].set_xlabel("Polynomial expansion degree")
    fig.suptitle(title, y=1.05)