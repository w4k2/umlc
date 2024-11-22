import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("foo.png")

print(X.shape)
print(np.unique(y))