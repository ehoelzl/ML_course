import numpy as np
from pca import perform_pca
from data_processing import normalize_columns

def remove_unnecessary_features(x):
    """Removes the columns (features) that have zero variance, and returns the index of kept columns"""
    std = np.std(x, axis=0)
    to_keep = np.where(std > 0)[0] # Columns to keep have non zero variance
    return x[:, to_keep], to_keep

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree (degree >=2)"""
    if degree < 2:
        return None
    tiled = np.tile(np.vstack(x), degree-1)
    poly = np.power(tiled, np.arange(2, degree+1))
    return poly

def polynomial_expansion(x, columns, degree):
    """Adds degrees to a given list of columns (appended)"""
    if degree < 2 :
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

def cosinus_expansion(x, columns):
    cos = np.cos(x[:, columns])
    return np.c_[x, cos]

def enhance_features(x, headers, poly_degree, pca_threshold=0.99, plot=True, proj_matrix=None, print_=True):
    """This function transforms the features of the given matrix by 
    - Eliminating columns with no variance
    - Normalizing the columns
    - Exponential, Logarithmic and polynomial expansion of all columns
    - Perform PCA or project on given projection matrix
    """
    x, cols = remove_unnecessary_features(x) # Removes features with no variance
    headers = headers[cols]
    if print_:
        print(f"The {len(headers)} features remaining after filtering zero-variance features are: \n\n {headers}\n")
    
    
    x = normalize_columns(x)
    if print_:
        print(f"Performing polynomial expansion up to degree {poly_degree}")
    
    #x = polynomial_expansion(x, np.arange(len(headers)), poly_degree)
    #x = exponential_expansion(x, np.arange(len(headers)))
    #x = logarithmic_expansion(x, np.arange(len(headers)))
    #x = sinus_expansion(x, np.arange(len(headers)))
    #x = cosinus_expansion(x, np.arange(len(headers)))
    
    if print_:
        print(f"Matrix has now {x.shape[1]} features")
    
    # So that we can use this function both to prepare for training and testing
    #x, proj_matrix = perform_pca(x, threshold=pca_threshold, plot=plot, proj_matrix=proj_matrix)
    return x, proj_matrix