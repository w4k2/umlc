from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from sklearn.base import clone
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from tabulate import tabulate

N_DATASETS = 30

RANDOM_STATE = 90210
drs = np.random.RandomState(42)

DATASET_SEEDS = drs.randint(100000, size=N_DATASETS)

DATASETS = []

for seed in DATASET_SEEDS:
    X, y = make_classification(n_samples=1000, random_state=seed)
    DATASETS.append([seed, (X, y)])

CLASSIFIERS = [
    ("NC_E", NearestCentroid(metric="euclidean")),
    ("NC_M", NearestCentroid(metric="manhattan")),
    ("KNN", KNeighborsClassifier()),
    ("MLP", MLPClassifier()),
]

SPLITS = []

splitter = RepeatedStratifiedKFold(random_state=RANDOM_STATE, n_repeats=5, n_splits=2)

dataset_scores = []

inconclusive = 0
first_better = 0
second_better = 0

for seed, (X, y) in tqdm(DATASETS):

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

scores = np.array(dataset_scores)

mean_scores = np.mean(scores, axis=1)

stats = np.zeros(shape=(len(CLASSIFIERS), len(CLASSIFIERS)))
ps = np.zeros(shape=(len(CLASSIFIERS), len(CLASSIFIERS)))

s, p = friedmanchisquare(*mean_scores.T)
print(s, p)

mean_ranks = np.mean(rankdata(mean_scores, axis=1), axis=0)
print(mean_ranks)

alpha = 0.05
alpha_corr = alpha / (len(CLASSIFIERS) - 1)

for i in range(len(CLASSIFIERS)):
    for j in range(len(CLASSIFIERS)):
        stat, p = wilcoxon(mean_scores[:, i], mean_scores[:, j])
        stats[i, j] = stat
        ps[i, j] = p

print(tabulate(stats))
print(tabulate(ps))
print(tabulate(ps < alpha_corr))
