import seaborn as sns
import matplotlib.pyplot as plt

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
    
def plot_label_distribution(x, y, headers, col_indices, title, ncols, nrows, figsize=(20, 10)):
    fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
    i, j = 0, 0
    for index in col_indices:
        if nrows == 1:
            axis = ax[j]
        elif ncols == 1:
            axis = ax[i]
        else:
            axis = ax[i][j]
        values = x[:, index]
        sns.scatterplot(x=values, y=y, ax=axis)
        axis.set_title(headers[index] + " " + str(index))
        axis.set_xlabel("Feature value")
        axis.set_ylabel("Label value")
        j = (j + 1) % ncols
        if j == 0:
            i = (i + 1) % nrows
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    
