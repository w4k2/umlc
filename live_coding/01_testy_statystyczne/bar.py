import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

split = RepeatedStratifiedKFold(n_splits=2, n_repeats=50)

X, y = load_iris(return_X_y=True)

scores_a =[]
scores_b = []

for train, test in split.split(X, y):
    clf = KNeighborsClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    score = accuracy_score(y[test], y_pred)
    scores_a.append(score)
    print('A', f"{score:.3f}")

    clf = GaussianNB()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    score = accuracy_score(y[test], y_pred)
    scores_b.append(score)
    print('B', f"{score:.3f}")

    print('--' * 20)

print(np.sum(np.array(scores_a) > np.array(scores_b)), '|', len(scores_a))