# %%
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# %%
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

classes = np.unique(y)

print(feature_names)
print(target_names)

# %%
plot_iris_hist()

# %%
plot_iris_kde()

# %%
plot_iris_scatter()

# %%

selected_features = (0, 1)
x = X[:, selected_features]

# %%

fig, ax = plt.subplots(1, 1, figsize=(1.5 * BW, 1.5 * BW))

ax.scatter(*x.T, c=colors[y])
ax.set_xlabel(feature_names[selected_features[0]])
ax.set_ylabel(feature_names[selected_features[1]])
ax.set_aspect(True)
ax.grid(True, ls=':')

plt.show()

# %%

centroids = np.array([
    np.mean(x[y == l], axis=0) for l in classes
])

# %%
fig, ax = plt.subplots(1, 1, figsize=(1.5 * BW, 1.5 * BW))

ax.scatter(*x.T, c=colors[y], alpha=0.6)

for c, center in zip(colors, centroids):
    ax.scatter(*center, marker="D", s=80, facecolor=c, edgecolors='k')

ax.set_xlabel(feature_names[selected_features[0]])
ax.set_ylabel(feature_names[selected_features[1]])
ax.set_aspect('equal')
ax.grid(True, ls=':')

plt.show()
# %%

from scipy.spatial.distance import cdist

def nearest_centroid(X, centroids):
    dist_mat = cdist(X, centroids)
    return np.argmin(dist_mat, axis=1)

# %%
fig, ax = plt.subplots(1, 1, figsize=(1.5 * BW, 1.5 * BW))

P = 100
xmin, ymin = np.min(x, axis=0)
xmax, ymax = np.max(x, axis=0)

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, P),
    np.linspace(ymin, ymax, P)
)

space = np.c_[xx.ravel(), yy.ravel()]

label = nearest_centroid(space, centroids)

# Plot space
ax.scatter(*space.T, c=colors[label], edgecolors='none', alpha=0.8)

for c, center in zip(colors, centroids):
    ax.scatter(*center, marker="D", s=80, facecolor=c, edgecolors='k')

ax.set_xlabel(feature_names[selected_features[0]])
ax.set_ylabel(feature_names[selected_features[1]])
ax.set_aspect('equal')
ax.grid(True, ls=':')

plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(1.5 * BW, 1.5 * BW))

# Plot space
ax.scatter(*space.T, c=colors[label], edgecolors='none', alpha=0.4)
for c, center in zip(colors, centroids):
    ax.scatter(*center, marker="D", s=80, facecolor=c, edgecolors='k')

# Plot misses
y_pred = nearest_centroid(x, centroids)
x_miss = y_pred != y
ax.scatter(*x[x_miss].T, edgecolors='k', facecolor=colors[y[x_miss]])

ax.set_xlabel(feature_names[selected_features[0]])
ax.set_ylabel(feature_names[selected_features[1]])
ax.set_aspect('equal')
ax.grid(True, ls=':')

plt.show()

# %%
total_samples = len(x)
accuracy = 1 - sum(x_miss) / total_samples
print("Example Accuracy:", accuracy)

# %%
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

clf = NearestCentroid()
clf.fit(X, y)
y_pred = clf.predict(X)

accuracy = accuracy_score(y, y_pred)

print("Iris Accuracy:", accuracy)


# %%
from sklearn.model_selection import RepeatedStratifiedKFold

acc_vec = []

for train, test in RepeatedStratifiedKFold().split(X, y):
    clf = NearestCentroid()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    accuracy = accuracy_score(y[test], y_pred)
    acc_vec.append(accuracy)

print("Accuracy mean:", np.mean(acc_vec))
print("Accuracy std:", np.std(acc_vec))