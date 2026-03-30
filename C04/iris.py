import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pca import PCA as PCA_

import numpy as np

X, y = load_iris(return_X_y=True)

X = StandardScaler().fit_transform(X)

fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(len(X.T)):
    for j in range(len(X.T)):
        if i == j:
            continue

        ax = axs[i, j]
        ax.scatter(X[:, i], X[:, j], c=y)
        ax.grid(ls=":")

plt.savefig("foo.png")

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

ax = axs[0]
pca_ = PCA_(k=2)
X_ = pca_.fit(X, y).transform(X)
ax.scatter(*X_.T, c=y)

ax = axs[1]

pca = PCA(n_components=2, svd_solver="covariance_eigh")
X_ = StandardScaler().fit_transform(X)
X_ = pca.fit(X_, y).transform(X_)
ax.scatter(*X_.T, c=y)

plt.savefig("bar.png")

fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(len(X.T)):
    for j in range(len(X.T)):
        if i == j:
            continue

        ax = axs[i, j]

        X_scaled = (X - pca_.mean_) / pca_.std_
        X_ = np.dot(X_scaled, pca_.eigenvectors_[:, [i, j]])

        ax.scatter(*X_.T, c=y)
        ax.grid(ls=":")

plt.savefig("baz.png")
