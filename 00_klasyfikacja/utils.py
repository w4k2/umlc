import numpy as np
import matplotlib.pyplot as plt

BW = 6

colors = np.array([
    "#0087FF",
    "#E63E00",
    "#3C8E0E",
])

from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

classes = np.unique(y)


def silverman_bandwidth(X):
    if X.ndim == 2:
        m, n = X.shape
    else:
        m, n = X.shape[0], 1
    
    return (m * (n + 2) / 4) ** (-1 / (n + 4))


def plot_iris_hist():
    fig, axs = plt.subplots(1, 4, figsize=(2 * BW, 0.5 * BW))

    for x_i , x in enumerate(X.T):
        ax = axs[x_i]

        for l in classes:
            ax.hist(x[y == l], alpha=0.8, color=colors[l], label=target_names[l])
            ax.set_xlabel(feature_names[x_i])
        
        ax.grid(True, ls=":")

    ax.legend()

    plt.show()

from scipy.stats import gaussian_kde

def plot_iris_kde():
    fig, axs = plt.subplots(1, 4, figsize=(2 * BW, 0.5 * BW))

    for x_i , x in enumerate(X.T):
        ax = axs[x_i]
        ls = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 100)

        for l in classes:
            bw = silverman_bandwidth(x[y == l])
            kde = gaussian_kde(x[y == l], bw_method=bw)
            ax.plot(ls, kde(ls), color=colors[l], label=target_names[l])
            ax.set_xlabel(feature_names[x_i])

        ax.grid(True, ls=":")

    ax.legend()

    plt.show()

from itertools import product

def plot_iris_scatter():
    fig, axs = plt.subplots(4, 4, figsize=(8, 8), dpi=60, sharex=True)

    X_min, X_max = np.min(X) - 0.5, np.max(X) + 0.5

    for i, j in product(range(4), range(4)):
        if i == j:
            x = X[:, i]
            ls = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 100)

            for c, l in zip(colors, classes):
                bw = silverman_bandwidth(x[y == l])
                kde = gaussian_kde(x[y == l], bw_method=bw)
#                 axs[i, j].set_title(feature_names[i])
                
                axs[i, j].plot(ls, kde(ls), color=c)
                axs[i, j].grid(ls=":")

            axs[i, j].grid(True, ls=":")
            continue

        x = X[:, (i, j)]
        for c, l in zip(colors, classes):
            axs[j, i].scatter(*x[y == l].T, c=c, s=5)
            axs[i, j].set_aspect("equal")
            axs[i, j].set_xlim(X_min, X_max)
            axs[i, j].set_ylim(X_min, X_max)
            axs[i, j].set_xlabel(feature_names[i])
            axs[i, j].set_ylabel(feature_names[j])
            axs[i, j].grid(ls=":")

    plt.tight_layout()
