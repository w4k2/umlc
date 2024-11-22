import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel
 
scores = np.load("scores.npy")

# tablefmt="latex"
table = tabulate(np.mean(scores, axis=-1), 
                 tablefmt="grid", 
                 headers=["KNN 3", "KNN 15", "RC"], 
                 showindex=["wisconsin", "iris"]
)

print(table)


result = ttest_rel(scores[0, 0, :], scores[0, 1, :])
print(result.statistic)
print(result.pvalue)
