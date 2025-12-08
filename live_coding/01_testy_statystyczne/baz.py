import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import shapiro

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers.random import RandomClassifier

# splits = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
splits = ShuffleSplit(n_splits=100, test_size=0.2)
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

s, a = shapiro(scores[:, 0])
print(f"Shapiro 1: (stat: {s:.2f} p-value: {a:.3f})")

s, a = shapiro(scores[:, 1])
print(f"Shapiro 2: (stat: {s:.2f} p-value: {a:.3f})")

s, a = ttest_rel(scores[:, 0], scores[:, 1])
print(f"t-test: (stat: {s:.2f} p-value: {a:.3f})")

s, a = wilcoxon(scores[:, 0], scores[:, 1], alternative="greater", method="exact")
print(f"wilcoxon: (stat: {s:.2f} p-valuex: {a:.32f})")

