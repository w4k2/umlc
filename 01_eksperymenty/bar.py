import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from random_classifier import RandomClassifier
from sklearn.neighbors import KNeighborsClassifier

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

X, y = load_breast_cancer(return_X_y=True)

print(X.shape)
print(np.unique(y, return_counts=True))

for train, test in rskf.split(X, y):
    clf = KNeighborsClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = accuracy_score(y[test], y_pred)
    print('+', score)