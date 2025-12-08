import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_prior=True, random_state=None):
        self.use_prior = use_prior
        self.random_state = random_state

        self._classes = None
        self._counts = None

    def fit(self, X, y):
        self._classes, self._counts = np.unique(y, return_counts=True)
        self._rs = np.random.RandomState(seed=self.random_state)
        return self

    def predict(self, X):
        p = self._counts / np.sum(self._counts) if self.use_prior else np.ones(len(self._classes)) / len(self._classes)
        labels = self._rs.choice(range(len(self._classes)), p=p, size=len(X))
        return self._classes.take(labels)
