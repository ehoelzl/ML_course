import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_feature_distribution(x, headers, col_indices, title, ncols, nrows):
    """Plots the distribution of each column of x"""
    fig, ax = plt.subplots(figsize=(20, 10), ncols=ncols, nrows=nrows)
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
    return fig


def plot_undefined_features(x, headers):
    """Plots the distribution of columns of x that have at least an undefined value (-999), and removes it"""
    fig, ax = plt.subplots(figsize=(20, 3), ncols=5, nrows=1)
    r, c = 0, 0
    col_ind = np.arange(x.shape[1])#np.where(np.min(x, axis=0) == -999)[0]
    for i in col_ind:
        
        t = x[:, i]
        t = t[np.where(t > -999)]
        sns.distplot(t, ax=ax[i])
        ax[i].set_title(headers[i])
        c = (c + 1) % 3
        if c == 0:
            r = (r + 1) % 4
            
    fig.suptitle("Distribution of PHI features", y=1.05)
    return fig
#    plt.tight_layout()


def correlation_plot(x, h):
    """Plots the correlation of each column of x"""
    corr = np.corrcoef(x.T)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(corr, cmap=sns.diverging_palette(200, 10, n=200), center=0,
                square=True, linewidths=.2, vmin=-1, vmax=1, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_xticks(np.arange(len(h)) + 0.5)
    ax.set_yticks(np.arange(len(h)) + 0.5)
    
    ax.set_xticklabels(h, rotation=90);
    ax.set_yticklabels(h, rotation=0);
    ax.set_ylim(-1, 31);
