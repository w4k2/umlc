import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class DumbClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_pred=0):
        self.class_pred = class_pred
        self._classes = None

    def fit(self, X, y):
        self._classes = np.unique(y)
        return self

    def predict(self, X):
        return self._classes.take(np.repeat(self.class_pred, len(X)))
