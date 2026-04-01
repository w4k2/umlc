from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from pca import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor

import numpy as np

data = load_digits()

X = data.data

X = X / 15

print(X.shape)
print(X[0])
y = data.target

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
img = X[126].reshape(8, 8)
img = img / 15
ax.imshow(img, cmap='grey', vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig("foo.png")

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
axs = axs.flatten()

for i in range(100):
    ax = axs[i]
    img = X[i].reshape(8, 8)
    ax.imshow(img, cmap='grey', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("foo.png")

y = data.target

split = StratifiedShuffleSplit(n_splits=1, random_state=200)
train, test = next(split.split(X, y))

clf = NearestCentroid()
clf.fit(X[train], y[train])
y_pred = clf.predict(X[test])

# score = accuracy_score(y[test], y_pred)
# print(score)
# exit()

score = confusion_matrix(y[test], y_pred)
print(score)

diag = np.diagonal(score)
total = np.sum(score, axis=1)
print(diag / total)

X_ = PCA(k=2).fit_transform(X)
fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y == l].T, label=str(l))

plt.legend()

plt.savefig("bar.png")


mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=1, verbose=True, learning_rate_init=0.001)
mlp.fit(X[train], y[train])
y_pred = mlp.predict(X[test])

print(balanced_accuracy_score(y[test], y_pred))


# fig, ax = plt.subplots(1, 1)

# X_ = X @ mlp.coefs_[0] + mlp.intercepts_[0]
# X_ = PCA(k=2).fit_transform(X_)

# for l in np.unique(y):
#     ax.scatter(*X_[train][y[train] == l].T, label=str(l), alpha=0.5)
#     ax.scatter(*X_[test][y[test] == l].T)

# plt.legend()
# plt.savefig("baz.png")
# plt.close()

# exit()



X_ = PCA(k=2).fit_transform(X_)

for _ in range(1000):
    mlp.partial_fit(X[train], y[train])
    fig, ax = plt.subplots(1, 1)

    X_ = X @ mlp.coefs_[0] + mlp.intercepts_[0]
    X_ = X_ @ mlp.coefs_[1] + mlp.intercepts_[1]

    for l in np.unique(y):
        ax.scatter(*X_[train][y[train] == l].T, label=str(l), alpha=0.5)
        ax.scatter(*X_[test][y[test] == l].T)

    plt.legend()
    plt.savefig("baz.png")
    plt.close()
# #
score = balanced_accuracy_score(y[test], mlp.predict(X[test]))
print(score)

X_ = PCA(k=2).fit_transform(X_)

fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y[test] == l].T, label=str(l))

plt.legend()
plt.savefig("baz.png")

y_pred = mlp.predict(X[test])
score = balanced_accuracy_score(y[test], y_pred)
print(score)


mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=10000, verbose=True)
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
