import os

import numpy as np

from itertools import combinations
from scipy.stats import ttest_rel
from tabulate import tabulate

from config import *

def stat_test_mat(series, alpha=0.05):
    s_mat = np.zeros((len(series), len(series)), dtype=bool)
    p_mat = np.zeros((len(series), len(series)), dtype=bool)

    # Check each pair
    for s1, s2 in combinations(range(len(series)), 2):
        s, p = ttest_rel(series[s1], series[s2])

        # Use p-value for evaluation
        p_mat[(s1, s2), (s2, s1)] = p < alpha

        # Use statistic to establish which mean is higher
        s_mat[s1, s2] = s > 0
        s_mat[s2, s1] = s < 0

    return s_mat, p_mat

def stat_results_to_list(s_mat, p_mat):
    combined_mat = np.logical_and(s_mat, p_mat)
    return [np.argwhere(row).flatten() for row in combined_mat]

def stat_lists_to_str(lists):
    n_all = len(lists) - 1

    def translate_row(a):
        if len(a) == n_all:
            return 'all'
        if len(a) == 0:
            return '--'
        else:
            return ', '.join(np.char.mod('%d', a + 1))

    return list(map(translate_row, lists))

results_cube = np.empty((len(METRICS), len(DATASETS), len(METHODS), SPLITS.get_n_splits()))

for ds_idx, ds_name in enumerate(DATASETS):
    for s_idx in range(N_SPLITS):
        ds = np.load(os.path.join(SPLITS_DIR, ds_name, f"{s_idx}.npz"))
        y_test = ds["y_test"]

        for clf_idx, clf_name in enumerate(METHODS):
            y_pred_path = os.path.join(PREDICTIONS_DIR, ds_name, str(s_idx), f"{clf_name}.npy")
            y_pred = np.load(y_pred_path)

            for metric in METRICS:
                metrics = METRICS[metric](y_test, y_pred)

            results_cube[:, ds_idx, clf_idx, s_idx] = metrics


for metric_name, metric_results in zip(METRICS, results_cube):
    print(f"{'+' * 40} {metric_name} {'+' * 40}")

    table = []
    for ds_name, dataset_results in zip(DATASETS, metric_results):

        mv = dataset_results.mean(axis=-1)
        mv = np.char.mod('%.3f', mv)

        sd = dataset_results.std(axis=-1)
        sd = np.char.mod('%.3f', sd)

        s, p = stat_test_mat(dataset_results)
        sl = stat_results_to_list(s, p)
        ss = stat_lists_to_str(sl)

        table_row = [f"{v} +/- {d}\n{s}" for v, d, s in zip(mv, sd, ss)]
        table.append([ds_name, *table_row])

    print(tabulate(table, headers=[f"{cn} ({i})" for i, cn in enumerate(METHODS, 1)], tablefmt="simple_grid"))

