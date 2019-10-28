import numpy as np


def remove_unnecessary_features(x, jet_col):
    """Removes the columns (features) that have zero variance, and returns the index of kept columns"""
    
    not_nan = [i for i in np.arange(x.shape[1]) if not np.all(np.isnan(x[:, i])) and i != jet_col]
    x = x[:, not_nan]
    std = np.nanstd(x, axis=0)
    x = x[:, np.where(std > 0)[0]]
    return x


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree (degree >=2)"""
    if degree < 2:
        return None
    tiled = np.tile(np.vstack(x), degree - 1)
    poly = np.power(tiled, np.arange(2, degree + 1))
    return poly


def polynomial_expansion(x, columns, degree):
    """Adds degrees to a given list of columns (appended)"""
    if degree < 2:
        return x
    
    for c in columns:
        poly = build_poly(x[:, c], degree)
        if poly is not None:
            x = np.c_[x, poly]
    return x


def exponential_expansion(x, columns):
    """Adds columns as exp() of given columns (appended)"""
    exp = np.exp(x[:, columns])
    return np.c_[x, exp]


def logarithmic_expansion(x, cols):
    """Adds columns as log(1 / (1+x)) and log(1+x) to all positive columns"""
    pos_columns = np.where(np.nanmin(x, axis=0) >= 0)[0]
    pos_columns = [i for i in pos_columns if i in cols]
    if len(pos_columns) == 0:
        return x
    
    inv_log = np.log(1 / (1 + x[:, pos_columns]))
    log = np.log(1 + x[:, pos_columns])
    return np.c_[x, inv_log, log]


def sinus_expansion(x, columns):
    sin = np.sin(x[:, columns])
    return np.c_[x, sin]


def cosine_expansion(x, columns):
    cos = np.cos(x[:, columns])
    return np.c_[x, cos]


def sqrt_expansion(x, columns):
    sqrt = np.sqrt(np.abs(x[:, columns]))
    return np.c_[x, sqrt]


def cross_multiply(x, columns):
    to_add = None
    done = []
    for c in columns:
        done.append(c)
        mult = np.multiply(np.vstack(x[:, c]), np.delete(x, done, axis=1))
        if to_add is None:
            to_add = mult
        else:
            to_add = np.c_[to_add, mult]
    if to_add is not None:
        x = np.c_[x, to_add]
    return x


def expand_features(x, poly_degree, jet_col, print_=True):
    """This function expands the features of the given matrix with:
     - Polynomial expansion
     - Exponential/Log
     - Sin /Cosine
     - Sqrt
    """
    x = remove_unnecessary_features(x, jet_col)
    
    if print_:
        print(f"Performing polynomial expansion up to degree {poly_degree}")
    col_num = x.shape[1]
    x = logarithmic_expansion(x, np.arange(col_num))
    x = sqrt_expansion(x, np.arange(col_num))
    x = polynomial_expansion(x, np.arange(col_num), poly_degree)
    x = sinus_expansion(x, np.arange(col_num))
    x = cosine_expansion(x, np.arange(col_num))
    
    if print_:
        print(f"Matrix has now {x.shape[1]} features")
    
    return x
