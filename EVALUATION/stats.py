from sklearn.neighbors import NearestCentroid

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import ttest_ind, ttest_rel
from ftest import F_test

from sklearn.base import clone
import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt

N_SPLITS = 10

drs = np.random.RandomState()
RANDOM_STATE = drs.randint(100000)
print(RANDOM_STATE)

CLASSIFIERS = [
    ("NC_E", NearestCentroid(metric="euclidean")),
    ("NC_M", NearestCentroid(metric="manhattan")),
]

splitter = RepeatedStratifiedKFold(random_state=RANDOM_STATE, n_splits=2, n_repeats=5)

X, y = make_classification(random_state=RANDOM_STATE, n_samples=200)

splits_scores = []
for split_idx, (train, test) in enumerate(splitter.split(X, y)):

    classifiers_scores = []

    for clf_name, clf_base in CLASSIFIERS:
        clf = clone(clf_base)
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score = balanced_accuracy_score(y[test], y_pred)

        classifiers_scores.append(score)

    splits_scores.append(classifiers_scores)

scores = np.array(splits_scores)

print(f"{np.mean(scores[:, 0]):.3f}, +/- ({np.std(scores[:, 0]):.3f})")
print(f"{np.mean(scores[:, 1]):.3f}, +/- ({np.std(scores[:, 1]):.3f})")
print(f'w/l/t: {np.sum(scores[:, 0] > scores[:, 1])} | {np.sum(scores[:, 0] < scores[:, 1])} | {np.sum(scores[:, 0] == scores[:, 1])}')



fig, ax = plt.subplots(1, 1, figsize=(7, 7))
space = np.linspace(0.5, 1, 400)
ax.grid(ls=":")

for results_i, results in enumerate(scores.T):
    mu = np.mean(results)
    sigma = np.sqrt(np.var(results))
    ax.scatter(results, np.repeat(-0.1 - 0.1 * results_i, len(results)), s=15, marker='x')
    pdf = stats.norm.pdf(space, mu, sigma)
    pdf = pdf / np.max(pdf)
    ax.plot(space, pdf, ls="--")

plt.savefig(f"stats.png")

s, p = F_test(scores[:, 0], scores[:, 1])
print(f'stat: {s:.3f}, p-value: {p:.3f}')
print('-' * 20)

