import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=100):
        self.random_state = random_state
        self._classes = None

    def fit(self, X, y):
        self._classes, counts = np.unique(y, return_counts=True)
        self._counts = counts / np.sum(counts)
        self._rs = np.random.RandomState(seed=self.random_state)
        return self

    def predict(self, X):
        labels = self._rs.choice(range(len(self._classes)), p=self._counts, size=len(X))
        return self._classes.take(labels)
