import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_distribution(x, headers, col_indices, title, ncols, nrows):
    fig, ax = plt.subplots(figsize=(20,10), ncols=ncols, nrows=nrows)
    i, j = 0, 0
    for index in col_indices:
        if nrows == 1:
            axis = ax[j]
        elif ncols == 1:
            axis = ax[i]
        else:
            axis = ax[i][j]
        values = x[:, index]
        sns.distplot(values, ax=axis)
        axis.set_title(headers[index] + " " + str(index))
        j = (j + 1) % ncols
        if j == 0:
            i = (i + 1) % nrows
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    
def plot_undefined_features(x, headers):
    fig, ax = plt.subplots(figsize=(15, 7), ncols=3, nrows=4)
    r, c = 0, 0
    col_ind = np.where(np.min(x, axis=0) == -999)[0]
    acc, h = [], []
    for i in col_ind:
        
        t = x[:, i]
        t = t[np.where(t > -999)]
        sns.distplot(t, ax=ax[r][c])
        ax[r][c].set_title(headers[i] + str(i))
        c = (c+1)%3
        if c == 0:
            r = (r+1)%4
    plt.tight_layout()