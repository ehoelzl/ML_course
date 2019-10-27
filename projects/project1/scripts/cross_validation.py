import numpy as np
from implementations import reg_logistic_regression, reg_logistic_regression_sgd, ridge_regression
from costs import compute_reg_logistic_loss_l2, compute_mse_loss
from proj1_helpers import compute_accuracy

from feature_expansion import expand_features, polynomial_expansion, remove_unnecessary_features
from data_processing import prepare_for_training
import matplotlib.pyplot as plt


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
    
    w, loss_tr = ridge_regression(y_train, x_train, lambda_)
    
    loss_val = compute_mse_loss(y_val, x_val, w) + (2 * lambda_ * np.linalg.norm(w) ** 2)
    
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc

def cross_validation_logistic(y, x, k_indices, k, lambda_, gamma, max_iters=500):
    """Performs one iteration of the k-fold cross validation using L2 Regularized Logistic regression"""
    val_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    x_val, y_val = x[val_indices], y[val_indices]
    x_train, y_train = x[train_indices], y[train_indices]
    
    initial_w = np.zeros((x.shape[1], 1))
    w, loss_tr = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma, _print=False)
    
    loss_val = compute_reg_logistic_loss_l2(y_val, x_val, w, lambda_)
                                         
    y_val[np.where(y_val == 0)] = -1
    acc = compute_accuracy(y_val, x_val, w)
    return w, loss_tr, loss_val, acc


# def cross_validate_reg(x, y, cols, gamma, lambdas, k_fold, title, max_iters=500):
#     x_train, y_train, x_test, y_test = prepare_for_training(x, y, cols, 0, split=False)
#
#     k_indices = build_k_indices(y, k_fold)
#     accuracies, losses = [], []
#     for lambda_ in lambdas:
#         accs, losses_val = [], []
#
#         for k in range(k_fold):
#             w, loss_tr, loss_val, acc = cross_validation_reg_logistic(y_train, x_train, k_indices, k, max_iters, gamma,
#                                                                       lambda_)
#
#             accs.append(acc)
#             losses_val.append(loss_val)
#         accuracies.append(accs)
#         losses.append(losses_val)
#
#     fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
#     ax[0].plot(np.mean(accuracies, axis=1))
#     ax[0].set_xticks(np.arange(len(lambdas)));
#     ax[0].set_xticklabels(lambdas);
#     ax[0].set_title("Mean validation accuracy")
#     ax[0].set_xlabel("Regularization paramter")
#
#     ax[1].plot(np.mean(losses, axis=1))
#     ax[1].set_xticks(np.arange(len(lambdas)));
#     ax[1].set_xticklabels(lambdas);
#     ax[1].set_title("Mean validation loss")
#     ax[1].set_xlabel("Regularization parameter")
#     fig.suptitle(title, y=1.05)


def cross_validate_degrees(x, y, lambda_, gamma, degrees, k_fold, title, logistic=True):
    
    accuracies, losses = [], []
    
    x, _ = remove_unnecessary_features(x)
    for deg in degrees:  # Iterate over each given degree
        x_new = polynomial_expansion(x, np.arange(x.shape[1]), deg)

        x_train, y_train, _, _ = prepare_for_training(x_new, y, None, 0, split=False,
                                                      logistic=logistic)  # Normalize/Standardize
        k_indices = build_k_indices(y_train, k_fold)
        accs, losses_val = [], []
        for k in range(k_fold):
            try:
                if logistic:
                    w, loss_tr, loss_val, acc = cross_validation_logistic(y_train, x_train, k_indices, k, lambda_, gamma)
                else:
                    w, loss_tr, loss_val, acc = cross_validation_ridge(y_train, x_train, k_indices, k, lambda_)
                accs.append(acc)
                losses_val.append(loss_val)
            except Exception as e:
                print(e)
        accuracies.append(accs)
        losses.append(losses_val)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(np.mean(accuracies, axis=1))
    ax.set_xticks(np.arange(len(degrees)));
    ax.set_xticklabels(degrees);
    ax.set_title(title)
    ax.set_xlabel("Polynomial expansion degree")
    ax.set_ylabel("Accuracy")
