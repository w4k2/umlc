import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

X, y = load_iris(return_X_y=True)

for train, test in rskf.split(X, y):
    clf = MLPClassifier(max_iter=10000, tol=0.1, random_state=100)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = accuracy_score(y[test], y_pred)
    print('+', score)