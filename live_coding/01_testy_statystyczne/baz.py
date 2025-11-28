import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from random_classifier import RandomClassifier

split = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

X, y = make_classification(
    n_samples=200, n_classes=2, flip_y=0, weights=(0.5, 0.5), random_state=100
)

classes, c_n = np.unique(y, return_counts=True)
print(classes, c_n)

scores = []

for train, test in split.split(X, y):
    clf = KNeighborsClassifier()

    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = balanced_accuracy_score(y[test], y_pred)
    scores.append(score)
    print(score)

print('+', np.mean(scores))
