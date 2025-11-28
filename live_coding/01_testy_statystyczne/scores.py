import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_ind

scores = np.load("scores.npy")

# tablefmt="latex"
table = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["Dataset", "3", "5", "11"],
    showindex=["wisconsin", "iris"],
)

print(table)

result = ttest_ind(scores[1, 1, :], scores[1, 2, :])
print(f"{result.statistic:.2f}")
print(f"{result.pvalue:.12f}")
