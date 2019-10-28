import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from costs import compute_mse_loss, compute_reg_logistic_loss_l2
from data_processing import prepare_for_training
from feature_expansion import expand_features
from implementations import reg_logistic_regression, ridge_regression, reg_logistic_regression_sgd
from proj1_helpers import compute_accuracy


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ridge(y, x, k_indices, k, lambda_):
    """Performs one iteration of the k-fold cross validation using L2 Regularized Logistic regression"""
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    x_val, y_val = prepare_for_training(x_val, y_val, logistic=False)
    x_train, y_train = prepare_for_training(x_train, y_train, logistic=False)
    
    w, loss_tr = ridge_regression(y_train, x_train, lambda_)
    
    loss_val = compute_mse_loss(y_val, x_val, w) + (2 * lambda_ * np.linalg.norm(w) ** 2)
    
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc


def cross_validation_logistic(y, x, k_indices, k, lambda_, gamma, max_iters=5000):
    """Performs one iteration of the k-fold cross validation using L2 Regularized Logistic regression"""
    # Separate train/val
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    # Prepare for testing/training
    x_val, y_val = prepare_for_training(x_val, y_val, logistic=True)
    x_train, y_train = prepare_for_training(x_train, y_train, logistic=True)
    
    initial_w = np.zeros((x_train.shape[1], 1))
    w, loss_tr = reg_logistic_regression_sgd(y_train, x_train, lambda_, initial_w, max_iters, gamma, _print=False)
    
    loss_val = compute_reg_logistic_loss_l2(y_val, x_val, w, lambda_)
    
    y_val[np.where(y_val == 0)] = -1  # Need to do that to compute accuracy
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc


def cross_validate_degrees(x, y, lambdas, gamma, degrees, k_fold, title, jet_col, logistic=True):
    fig, ax = plt.subplots(figsize=(12, 15), ncols=2, nrows=len(lambdas))
    # x = remove_unnecessary_features(x, jet_col)  # Remove columns that are un-necessary
    for i, lambda_ in tqdm(enumerate(lambdas)):
        accuracies, losses = [], []
        
        for deg in degrees:  # Iterate over each given degree
            x_new = expand_features(x, deg, jet_col, print_=False)
            
            k_indices = build_k_indices(y, k_fold)
            accs, losses_val = [], []
            for k in range(k_fold):
                try:
                    if logistic:
                        w, loss_tr, loss_val, acc = cross_validation_logistic(y, x_new, k_indices, k, lambda_, gamma)
                    else:
                        w, loss_tr, loss_val, acc = cross_validation_ridge(y, x_new, k_indices, k, lambda_)
                    accs.append(acc)
                    losses_val.append(loss_val)
                except Exception as e:
                    print(e)
            accuracies.append(accs)
            losses.append(losses_val)
        
        ax[i][0].plot(np.mean(accuracies, axis=1))
        ax[i][0].set_xticks(np.arange(len(degrees)));
        ax[i][0].set_xticklabels(degrees);
        ax[i][0].set_title(f"lambda={lambda_}")
        ax[i][0].set_xlabel("Polynomial expansion degree")
        ax[i][0].set_ylabel("Accuracy")
        
        ax[i][1].plot(np.mean(losses, axis=1))
        ax[i][1].set_xticks(np.arange(len(degrees)));
        ax[i][1].set_xticklabels(degrees);
        ax[i][1].set_title(f"lambda={lambda_}")
        ax[i][1].set_xlabel(f"Polynomial expansion degree")
        ax[i][1].set_ylabel("Loss")
        plt.tight_layout()
        fig.suptitle(title, y=1.05)
    return fig
