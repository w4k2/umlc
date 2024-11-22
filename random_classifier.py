import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state):
        self.random_state = random_state

    def fit(self, X, y):
        self._classes = np.unique(y)
        self._rs = np.random.RandomState(seed=self.random_state)
        return self

    def predict(self, X):
        labels = self._rs.randint(0, len(self._classes), len(X))
        return self._classes.take(labels)
