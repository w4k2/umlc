import numpy as np

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter

from tqdm import tqdm

# splits = ShuffleSplit(n_splits=10, test_size=0.2)
splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
X, y = make_classification(n_samples=500, flip_y=0, weights=[0.99, 0.01], return_X_y=True)


stat_results = []

for n in tqdm(range(100)):
    scores = []

    for train, test in splits.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=9)
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score_1 = balanced_accuracy_score(y[test], y_pred)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score_2 = balanced_accuracy_score(y[test], y_pred)

        scores.append([score_1, score_2])

    scores = np.array(scores)

    s, a = ttest_rel(scores[:, 0], scores[:, 1])

    ttest_result = 0
    if a < 0.05:
        ttest_result = 1 if s > 0 else -1

    s, a = wilcoxon(scores[:, 0], scores[:, 1], alternative="greater", method="exact")

    wcx_result = 1 if a < 0.05 else 0
    if not wcx_result:
        s, a = wilcoxon(scores[:, 1], scores[:, 0], alternative="greater", method="exact")
        wcx_result = -1 if a < 0.05 else 0

    stat_results.append([ttest_result, wcx_result])

stat_results = np.array(stat_results)

print("t-test", Counter(stat_results[:, 0]))
print("wilcoxon", Counter(stat_results[:, 1]))