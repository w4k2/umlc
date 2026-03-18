from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from scipy.spatial.distance import cdist
import numpy as np

class NearestCentroidEx(BaseEstimator, ClassifierMixin):
    def __init__(self, p=2, center_function="mean"):
        self.p = p
        self.center_function = center_function

        self._classes = None
        self._centroids = None

    def fit(self, X, y):
        self._classes = np.unique(y)

        if self.center_function == "median":
            self._centroids = np.array([
                np.median(X[y == l], axis=0) for l in self._classes
            ])
        elif self.center_function == "mean":
            self._centroids = np.array([
                np.mean(X[y == l], axis=0) for l in self._classes
            ])
        else:
            raise ValueError(f"{self.center_function} not in ['median', 'mean']")

        return self

    def predict(self, X):
        dist_mat = cdist(X, self._centroids, metric='minkowski', p=2)
        return self._classes.take(np.argmin(dist_mat, axis=1))
