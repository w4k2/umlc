from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from scipy.spatial.distance import cdist
import numpy as np

class NearestCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self, p=2):
        self.p = p

        self.classes_ = None
        self.centroids_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self.centroids_ = np.array([
            np.mean(X[y == l], axis=0) for l in self.classes_
        ])

        return self

    def predict(self, X):
        dist_mat = cdist(X, self.centroids_, metric='minkowski', p=self.p)
        return self.classes_.take(np.argmin(dist_mat, axis=1))
