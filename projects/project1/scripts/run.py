from proj1_helpers import *
import numpy as np
from data_processing import split_dataset, prepare_for_training, prepare_for_testing
from implementations import ridge_regression, reg_logistic_regression
from feature_expansion import expand_features

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../data/submission.csv'

JET_COL = 18
MAX_ITERS = 1200
GAMMA = 1e-5

def predict_testset(x, degrees, w):
    x = expand_features(x, degrees, print_=False, jet_col=JET_COL)
    x, _ = prepare_for_testing(x, None, logistic=False)
    y_pred = predict_labels(w, x)
    
    return y_pred


def train_ridge_model(x, y, lambda_):
    x_train, y_train = prepare_for_training(x, y, logistic=False)
    
    weights, loss = ridge_regression(y_train, x_train, lambda_)
    
    return weights


def train_logistic_model(x, y, gamma, lambda_, max_iters):
    x_train, y_train = prepare_for_training(x, y)
    
    initial_w = np.zeros((x_train.shape[1], 1))
    weights, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma, _print=False)
    
    return weights


def main():
    np.random.seed(1)
    
    y, tX, ids, _ = load_csv_data(DATA_TRAIN_PATH)
    
    # Prepare feature matrix
    to_delete = [9, 15, 18, 20, 25, 28]
    to_keep = [x for x in np.arange(tX.shape[1]) if x not in to_delete]
    tX = tX[:, to_keep]
    tX[tX == -999] = np.nan
    
    # Split the dataset
    tX_0, y_0, tX_1, y_1, tX_2, y_2 = split_dataset(tX, y, jet_col=JET_COL)  # Split into each category
    
    # Train each category separately
    degree_0 = 1
    lambda_0 = 1e-2
    tX_0_exp = expand_features(tX_0, degree_0, jet_col=JET_COL, print_=False)
    print(f"Training regularized logistic model with lambda={lambda_0}, polynomial expansion up to {degree_0}")
    w_0 = train_logistic_model(tX_0_exp, y_0, GAMMA, lambda_0, MAX_ITERS)
    
    degree_1 = 1
    lambda_1 = 1e-2
    tX_1_exp = expand_features(tX_1, degree_1, jet_col=JET_COL, print_=False)
    print(f"Training regularized logistic model with lambda={lambda_1}, polynomial expansion up to {degree_1}")
    w_1 = train_logistic_model(tX_1_exp, y_1, GAMMA, lambda_1, MAX_ITERS)
    
    degree_2 = 1
    lambda_2 = 1e-2
    tX_2_exp = expand_features(tX_2, degree_2, jet_col=JET_COL, print_=False)
    print(f"Training regularized logistic model with lambda={lambda_2}, polynomial expansion up to {degree_2}")
    
    w_2 = train_logistic_model(tX_2_exp, y_2, GAMMA, lambda_2, MAX_ITERS)
    
    # Load test set and predict on each category
    _, tX_test, ids_test, _ = load_csv_data(DATA_TEST_PATH)
    tX_test = tX_test[:, to_keep]
    tX_test[tX_test == -999] = np.nan
    
    tX_0_test, ids_0, tX_1_test, ids_1, tX_2_test, ids_2 = split_dataset(tX_test, ids_test, jet_col=JET_COL)
    
    y_pred_0 = predict_testset(tX_0_test, degree_0, w_0)
    y_pred_1 = predict_testset(tX_1_test, degree_1, w_1)
    y_pred_2 = predict_testset(tX_2_test, degree_2, w_2)
    ids_test = np.concatenate([ids_0, ids_1, ids_2])
    y_pred = np.concatenate([y_pred_0, y_pred_1, y_pred_2])
    
    assert y_pred.shape[0] == tX_test.shape[0]
    
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
