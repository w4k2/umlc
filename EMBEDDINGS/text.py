from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# from pca import PCA
from sklearn.decomposition import PCA

data = fetch_20newsgroups()

X = np.array(data.data)
y = data.target
y_names = data.target_names

y_sel = (0, 1, 2, 3, 4, 5)
mask = np.isin(y, y_sel).flatten()

X = X[mask]
y = y[mask]

# print(len(X))

# print(X[0])
# print(y_names[y[0]])
# print(y_names)
# print(len(y_names))

X_vec = TfidfVectorizer().fit_transform(X)

# X_ = PCA(n_components=2).fit_transform(X_vec)

# fig, ax = plt.subplots(1, 1)

# for l in np.unique(y):
#     ax.scatter(*X_[y == l].T, label=y_names[l])

# plt.legend()
# plt.savefig("foo.png")

# exit()

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, verbose=True)
mlp.fit(X_vec, y)

X_ = X_vec @ mlp.coefs_[0] + mlp.intercepts_[0]
X_ = PCA(n_components=2).fit_transform(X_)

fig, ax = plt.subplots(1, 1)

for l in np.unique(y):
    ax.scatter(*X_[y == l].T, label=y_names[l])

plt.legend()
plt.savefig("bar.png")
