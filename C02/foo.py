from sklearn.neighbors import NearestCentroid
from nearest_centroid import NearestCentroidEx

from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score

from sklearn.base import clone
import numpy as np

sklearn_clf = NearestCentroid()
um_clf = NearestCentroidEx(center_function='median')


X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, flip_y=0)

splitter = ShuffleSplit(random_state=100, n_splits=1, test_size=0.2)

train, test = next(splitter.split(X, y))

sklearn_clf.fit(X[train], y[train])
sk_y_pred = sklearn_clf.predict(X[test])
sklearn_acc = accuracy_score(y[test], sk_y_pred)
print("sklearn", f"{sklearn_acc:.5f}")

um_clf.fit(X[train], y[train])
um_y_pred = um_clf.predict(X[test])
um_acc = accuracy_score(y[test], um_y_pred)
print("UM", f"{um_acc:.5f}")


um_clf_copy = clone(um_clf)
try:
    um_clf_copy.predict(X[test])
    print("[ok]")
except Exception as e:
    print(f"[not ok] {e}")


import matplotlib.pyplot as plt

colors = np.array(['r', 'b'])

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

ax = axs[0]
ax.scatter(*X[train].T, c=colors[y[train]], alpha=0.2)
ax.scatter(*X[test].T, c=colors[sk_y_pred], edgecolors=colors[y[test]], linewidths=2, alpha=0.6)

C = sklearn_clf.centroids_
print(C)

for c, color in zip(C, colors):
    ax.scatter(*c, c=color, marker='x', s=80)

ax.set_aspect('equal')


ax = axs[1]
ax.scatter(*X[train].T, c=colors[y[train]], alpha=0.2)
ax.scatter(*X[test].T, c=colors[um_y_pred], edgecolors=colors[y[test]], linewidths=2, alpha=0.6)

C = um_clf._centroids
print(C)

for c, color in zip(C, colors):
    ax.scatter(*c, c=color, marker='x', s=80)

ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("foo.png")
plt.close()