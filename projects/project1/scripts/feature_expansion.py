import numpy as np


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


def logarithmic_expansion(x, columns):
    """Adds columns as log(abs() + 1) of give columns"""
    log = np.log(np.abs(x[:, columns]) + 1)
    return np.c_[x, log]


def sinus_expansion(x, columns):
    sin = np.sin(x[:, columns])
    return np.c_[x, sin]


def cosine_expansion(x, columns):
    cos = np.cos(x[:, columns])
    return np.c_[x, cos]


def sqrt_expansion(x, columns):
    sqrt = np.sqrt(np.abs(x[:, columns]))
    return np.c_[x, sqrt]


def expand_features(x, poly_degree, print_=True):
    """This function expands the features of the given matrix with:
     - Polynomial expansion
     - Exponential/Log
     - Sin /Cosine
     - Sqrt
    """
    
    if print_:
        print(f"Performing polynomial expansion up to degree {poly_degree}")
    
    col_num = x.shape[1]
    
    x = polynomial_expansion(x, np.arange(col_num), poly_degree)
    x = exponential_expansion(x, np.arange(col_num))
    x = logarithmic_expansion(x, np.arange(col_num))
    x = sinus_expansion(x, np.arange(col_num))
    x = cosine_expansion(x, np.arange(col_num))
    x = sqrt_expansion(x, np.arange(col_num))
    
    if print_:
        print(f"Matrix has now {x.shape[1]} features")
    
    return x
