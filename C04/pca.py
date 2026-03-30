import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, k=2) -> None:
        self.k = k

        self.mean_ = None
        self.std_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Standardize the data manually
        X_scaled = np.nan_to_num((X - self.mean_) / self.std_)

        # Calculate the Covariance Matrix
        cov_matrix = np.cov(X_scaled, rowvar=False)

        # Calculate Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        return self

    def transform(self, X, y=None):
        # 6. Select the top 'k' eigenvectors (our Principal Components)
        sorted_index = np.argsort(self.eigenvalues_)[::-1]
        sorted_eigenvectors = self.eigenvectors_[:, sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, 0:self.k]

        X_scaled = np.nan_to_num((X - self.mean_) / self.std_)

        # Make Projection
        return X_scaled @ eigenvector_subset
