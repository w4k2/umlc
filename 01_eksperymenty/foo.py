import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from random_classifier import RandomClassifier
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import ttest_rel

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
X, y = load_breast_cancer(return_X_y=True)

print('=' * 30)

c_labels, c_n = np.unique(y, return_counts=True)

print("samples:", X.shape[0])
print("features:", X.shape[1])
print("classes:", len(c_labels))
print("class count:", c_n)
print("IR:", np.max(c_n) / np.min(c_n))

print('=' * 30)

scores_knn = []
scores_rc = []

for s_i, (train, test) in enumerate(rskf.split(X, y)):
    clf = KNeighborsClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = accuracy_score(y[test], y_pred)
    scores_knn.append(score)
    print(s_i, 'KNN:', score)

    clf = RandomClassifier()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    score = accuracy_score(y[test], y_pred)
    scores_rc.append(score)
    print(s_i, 'RC:', score)

    print('*' * 30)

print('=' * 30)
print(f"KNN {np.mean(scores_knn):.2f} ({np.std(scores_knn):.2f})")
print(f"KNN {np.mean(scores_rc):.2f} ({np.std(scores_rc):.2f})")

t, p = ttest_rel(scores_knn, scores_rc)

print(f"stat: t: {t}, p: {p} | p > 0.05: {p < 0.05}")