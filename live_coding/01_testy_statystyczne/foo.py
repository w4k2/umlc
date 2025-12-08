import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from classifiers.dumb import DumbClassifier
from classifiers.random import RandomClassifier

# X, y = load_iris(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)
X, y = make_classification(n_samples=500, flip_y=0, weights=[0.99, 0.01])

print("=" * 20)
print("Features", X.shape[1])
print("Samples", X.shape[0])
c, class_samples = np.unique(y, return_counts=True)
print("Classes", c)
print("Sample Ratio", class_samples / np.sum(class_samples))
print("=" * 20)

splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train, test in splits.split(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    print(f"KNN Accuracy: {f1_score(y[test], y_pred):.3f}")

    clf = RandomClassifier(use_prior=False)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    print(f"RC (NP) Accuracy: {f1_score(y[test], y_pred):.3f}")

    clf = RandomClassifier(use_prior=True)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    print(f"RC (P) Accuracy: {f1_score(y[test], y_pred):.3f}")

    clf = DumbClassifier(class_pred=0)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    print(f"DC (0) Accuracy: {f1_score(y[test], y_pred):.3f}")

    clf = DumbClassifier(class_pred=1)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    print(f"DC (1) Accuracy: {f1_score(y[test], y_pred):.3f}")
