import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers.random import RandomClassifier
from classifiers.dumb import DumbClassifier

splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

X, y = make_classification(n_samples=500, flip_y=0, weights=[0.99, 0.01], return_X_y=True)

scores = []

for train, test in splits.split(X, y):
    clf = KNeighborsClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score_1 = accuracy_score(y[test], y_pred)

    clf = DecisionTreeClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score_2 = accuracy_score(y[test], y_pred)

    scores.append([score_1, score_2])

scores = np.array(scores)

for i, (mean, std) in enumerate(zip(np.mean(scores, axis=0), np.std(scores, axis=0))):
    print(f"{i}: {mean:.2f} ({std:.2f})")

print('.' * 20)
print("wins:", np.sum(scores[:, 0] > scores[:, 1]))
print("ties:", np.sum(scores[:, 0] == scores[:, 1]))
print("looses:", np.sum(scores[:, 0] < scores[:, 1]))

print('-' * 20)