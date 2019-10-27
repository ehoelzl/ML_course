import numpy as np
import matplotlib.pyplot as plt
from data_processing import standardize_columns


def remove_unnecessary_features(x):
    """Removes the columns (features) that have zero variance, and returns the index of kept columns"""
    std = np.std(x, axis=0)
    to_keep = [x for x in np.where(std > 0)[0] if x != 22]  # Columns to keep have non zero variance
    return x[:, to_keep], to_keep


def pca(x):
    corr_matrix = np.corrcoef(x.T)  # Compute correlation matrix
    eig_vals, eig_vecs = np.linalg.eig(corr_matrix)  # eigen value decomposition
    
    for ev in eig_vecs.T:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))  # Check that eigen vectors have norm of 1
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), np.real(eig_vecs[:, i])) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    return eig_pairs, var_exp, cum_var_exp


def pca_plot(var_exp, cum_var_exp):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 4))
        
        plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()


def perform_pca(x, headers, threshold, plot, proj_matrix, print_=False):
    """Function that reduces the features of a matrix to K. It can work with either a given projection matrix, or it can compute one
    :param x: Feature matrix (N,D)
    :param headers: The headers of the initial matrix
    :param threshold: The threshold for cumulative variance (determines # of features)
    :param plot: Whether to plot the result of PCA (works inly if proj_matrix=None)
    :param proj_matrix: Either a matrix (D,K) or None
    :return: The projected x matrix
    """
    x, cols = remove_unnecessary_features(x)  # Removes features with no variance
    headers = headers[cols]
    if print_:
        print(f"The {len(headers)} features remaining after filtering zero-variance features are: \n\n {headers}\n")

    x_std = standardize_columns(x)  # Standardize columns before PCA
    
    if proj_matrix is None:
        eig_pairs, var_exp, cum_var_exp = pca(x_std)
        index = np.where(cum_var_exp > threshold)[0][0]
        if plot:
            print(f"We can see that {index} features explain {threshold * 100}% of the cumulative variance")
            pca_plot(var_exp, cum_var_exp)
        
        proj_matrix = np.hstack([np.vstack(eig_pairs[i][1]) for i in range(index)])  # Projection matrix
    
    x_new = x_std.dot(proj_matrix)
    return x_new, proj_matrix
