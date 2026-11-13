from sklearn.neighbors import NearestCentroid

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import ttest_rel, ttest_ind
from ftest import F_test

from sklearn.base import clone
import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt

from tqdm import tqdm

N_SEEDS = 1000

RANDOM_STATE = 90210
X, y = make_classification(random_state=RANDOM_STATE, n_samples=200)

drs = np.random.RandomState(RANDOM_STATE)
SEEDS = drs.randint(100000, size=N_SEEDS)

CLASSIFIERS = [
    ("NC_E", NearestCentroid(metric="euclidean")),
    ("NC_M", NearestCentroid(metric="manhattan")),
]

inconclusive = 0
first_better = 0
second_better = 0

for seed in SEEDS:
    splitter = StratifiedShuffleSplit(random_state=seed, n_splits=10, test_size=0.2)

    splits_scores = []

    for split_idx, (train, test) in enumerate(splitter.split(X, y)):

        classifiers_scores = []

        for (clf_name, clf_base) in CLASSIFIERS:
            clf = clone(clf_base)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score = balanced_accuracy_score(y[test], y_pred)

            classifiers_scores.append(score)

        splits_scores.append(classifiers_scores)

    scores = np.array(splits_scores)
    s, p = ttest_rel(scores[:, 0], scores[:, 1])

    if p >= 0.05:
        inconclusive += 1
    elif np.mean(scores[:, 0]) > np.mean(scores[:, 1]):
        first_better += 1
    else:
        second_better += 1

    print(f"= {seed} =")
    print(f"{np.mean(scores[:, 0]):.3f}, {np.std(scores[:, 0]):.3f}")
    print(f"{np.mean(scores[:, 1]):.3f}, {np.std(scores[:, 1]):.3f}")
    print(f'w/l/t: {np.sum(scores[:, 0] > scores[:, 1])} | {np.sum(scores[:, 0] < scores[:, 1])} | {np.sum(scores[:, 0] == scores[:, 1])}')
    print(f'stat: {s:.3f}, p-value: {p:.3f}')
    print('-' * 20)

print("+" * 100)
print(f"[first, second, inconclusive]: {first_better} | {second_better} | {inconclusive}")
