import numpy as np

from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# from classifiers.nearest_centroid import NearestCentroid
from classifiers.dummy_classifier import DumbClassifier
# from sklearn.neighbors import NearestCentroid

import matplotlib.pyplot as plt

RANDOM_STATE = 42

X, y = make_classification(random_state=RANDOM_STATE, n_samples=1000, flip_y=0.0, weights=[0.99, 0.01])

# print(np.unique(y, return_counts=True))

# exit()
# fig, ax = plt.subplots(20, 20, figsize=(100, 100))

# for i in range(20):
#     for j in range(20):
#         ax[i, j].scatter(*X[:, (i, j)].T, c=y)

# plt.savefig('foo.png')

# X, y = load_iris(return_X_y=True)

# splits = ShuffleSplit(n_splits=100, test_size=0.2)
splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=20)

scores = []

for train, test in splits.split(X, y):
    clf = DumbClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = balanced_accuracy_score(y[test], y_pred)
    scores.append(score)

print(np.mean(scores), np.std(scores))
