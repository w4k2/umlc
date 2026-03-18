from random_classifier import RandomClassifier
from dummy_classifier import DumbClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score



X, y = make_classification(n_samples=200, flip_y=0, weights=[0.8, 0.2])
splitter = ShuffleSplit(random_state=100, n_splits=1, test_size=0.5)

train, test = next(splitter.split(X, y))

clf = RandomClassifier()

clf.fit(X[train], y[train])
y_pred = clf.predict(X[test])

score = accuracy_score(y[test], y_pred)
print(score)

score = balanced_accuracy_score(y[test], y_pred)
print(score)