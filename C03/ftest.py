from sklearn.neighbors import NearestCentroid

from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import ttest_rel
from scipy import stats
from sklearn.base import clone
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm


def F_test(a, b, cv=(5, 2)):
    n_repeats, k_folds = cv

    delta = a.reshape(n_repeats, k_folds) - b.reshape(n_repeats, k_folds)
    fold_mean = np.mean(delta, axis=-1)
    var = np.sum(np.power(delta - fold_mean[:, None], 2), axis=-1)

    f_stat = np.sum(np.power(delta, 2)) / (2 * np.sum(var))
    pvalue = stats.f.sf(f_stat, k_folds, k_folds * n_repeats)

    return f_stat, pvalue

N_SPLITS = 10
N_DATASETS = 10000

RANDOM_STATE = 90210
drs = np.random.RandomState(42)

DATASET_SEEDS = drs.randint(100000, size=N_DATASETS)

print(DATASET_SEEDS)

DATASETS = []

for seed in DATASET_SEEDS:
    X, y = make_classification(n_samples=1000, random_state=seed)
    DATASETS.append([seed, (X, y)])
    break

CLASSIFIERS = [
    ("NC_E", NearestCentroid(metric="euclidean")),
    ("NC_M", NearestCentroid(metric="manhattan")),
]

SPLITS = []

for seed in DATASET_SEEDS:
    splitter = RepeatedStratifiedKFold(random_state=seed, n_repeats=5, n_splits=2)
    SPLITS.append([seed, splitter])

dataset_scores = []

_, (X, y) = DATASETS[0]

inconclusive = 0
first_better = 0
second_better = 0

for rep_idx, (seed, splitter) in enumerate(tqdm(SPLITS)):

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

    dataset_scores.append(splits_scores)

    scores = np.array(splits_scores)
    delta = np.mean(scores[:, 0]) - np.mean(scores[:, 1])
    s, p = F_test(scores[:, 0], scores[:, 1])

    if p > 0.05:
        inconclusive += 1
    elif delta > 0:
        first_better += 1
    else:
        second_better += 1

    if rep_idx > 9:
        continue

    print(f"= {seed} =")
    print(f"{np.mean(scores[:, 0]):.3f}, {np.std(scores[:, 0]):.3f}")
    print(f"{np.mean(scores[:, 1]):.3f}, {np.std(scores[:, 1]):.3f}")
    print(f'+/-: {np.sum(scores[:, 0] > scores[:, 1])} | {np.sum(scores[:, 0] < scores[:, 1])} | {np.sum(scores[:, 0] == scores[:, 1])}')
    print(f'stat: {s:.3f}, p-value: {p:.3f}')
    print('-' * 20)

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

    plt.savefig(f"ftest_{seed}.png")
    plt.close()

print(f"{first_better} | {second_better} | {inconclusive}")
