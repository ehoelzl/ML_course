import numpy as np

def split_dataset(x, y, jet_col):
    """Splits the dataset into 3 categories (0 jets, 1 jet and more than 1 jet)
    :param x: Feature matrix (initial)
    :param y: Labels
    :param jet_col: Column of jet category (default=22)
    :return: x_0, y_0, x_1, y_1, x_2, y_2 The feature matrix and labels for each category
    """
    jets_0_inds, jets_1_inds, jets_rest_inds = np.where(x[:, jet_col] == 0), np.where(x[:, jet_col] == 1), np.where(x[:, jet_col] > 1)
    x_0, y_0 = x[jets_0_inds], y[jets_0_inds]
    x_1, y_1 = x[jets_1_inds], y[jets_1_inds]
    x_more, y_more = x[jets_rest_inds], y[jets_rest_inds]
    return x_0, y_0, x_1, y_1, x_more, y_more


def get_feature_columns(x):
    """Returns the indices of columns that have non null variance (i.e. columns where features vary)
    :param x: Feature matrix
    :return: np.array List of indices where the variance is > 0
    """
    std = np.std(x, axis=0)
    return np.where(std > 0)[0]

def standardize_matrix_columns(x):
    """Standardizes the columns of the matrix"""
    means = np.mean(x, axis=0)
    x = x - means
    std = np.std(x, axis=0)
    return x / std
    
def prepare_dataset(tx, y, headers):
    features = get_feature_columns(tx) # Returns the column numbers that have variance (discards all the rest)
    new_headers = headers[features] # Keep only the headers of those features
    tx = tx[:, features]  # Keep only interesting columns
    tx = standardize_matrix_columns(tx)  #Standardize
    
    tx = np.c_[np.ones((y.shape[0], 1)), tx] # Add bias 
    new_headers = np.insert(new_headers, 0, ["Bias"]) # Add bias column
    y = np.vstack(y)
    
    return tx, y, new_headers
    
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree (degree >=2)"""
    if degree < 2:
        return x
    tiled = np.tile(np.vstack(x), degree-1)
    poly = np.power(tiled, np.arange(2, degree+1))
    return poly


def add_degrees(x, column, degree, col_names):
    """Given the whole feature matrix, this functions builds a polynomial fo the given column number and adds it to the matrix
    :param x: Feature matrix
    :param column: Column number to add polynomial
    :param degree: The maximum degree of the polynomial (>=2)
    :param col_names: The name of initial columns (adds the powers to the name and retuns the new column names)
    :return: X (new feature matrix with added columns), new_col_names
    """
    if degree < 2:
        return x, col_names
    
    poly = build_poly(x[:, column], degree)
    
    cname = col_names[column]
    new_cols = [f"{cname}**{i}" for i in np.arange(2, degree+1)]
    new_col_names = np.insert(col_names.astype(object), column + 1, new_cols)
    return np.c_[x[:, :column+1], poly, x[:, column+1:]], new_col_names


def add_degrees_to_columns(x, headers, columns, degree):
    """Adds degrees to a given list of columns"""
    for i, c in enumerate(columns):
        col = c + (degree -1) * i
        x, headers = add_degrees(x, col, degree, headers)
    return x, headers