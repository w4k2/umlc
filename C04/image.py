from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from pca import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor

import numpy as np

data = load_digits()

X = data.data

X = X / 15

print(X.shape)

# fig, axs = plt.subplots(10, 10, figsize=(10, 10))
# axs = axs.flatten()

# for i in range(100):
#     ax = axs[i]
#     img = X[i].reshape(8, 8)
#     ax.imshow(img, cmap='grey', vmin=0, vmax=1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.tight_layout()
#     plt.savefig("foo.png")


y = data.target

print(y)
print(y.shape)

split = StratifiedShuffleSplit(n_splits=1, random_state=200)

train, test = next(split.split(X, y))

clf = NearestCentroid()
clf.fit(X[train], y[train])
y_pred = clf.predict(X[test])

score = balanced_accuracy_score(y[test], y_pred)
print(score)

X_ = PCA(k=2).fit_transform(X)

fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y == l].T, label=str(l))

plt.legend()

plt.savefig("bar.png")

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)
mlp.fit(X[train], y[train])

X_ = X[test] @ mlp.coefs_[0] + mlp.intercepts_[0]
X_ = PCA(k=2).fit_transform(X_)

fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y[test] == l].T, label=str(l))

plt.legend()
plt.savefig("baz.png")

y_pred = mlp.predict(X[test])
score = balanced_accuracy_score(y[test], y_pred)
print(score)


mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000)

mlp.fit(X, X)
X_pred = mlp.predict(X)

# fig, axs = plt.subplots(10, 10, figsize=(10, 10))
# axs = axs.flatten()

# for i in range(100):
#     ax = axs[i]
#     img = X_pred[i].reshape(8, 8)
#     ax.imshow(img, cmap='grey', vmin=0, vmax=1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.tight_layout()
#     plt.savefig("foo2.png")

X_ = X @ mlp.coefs_[0] + mlp.intercepts_[0]
X_ = PCA(k=2).fit_transform(X_)

fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y == l].T, label=str(l))

plt.legend()
plt.savefig("baz2.png")
