from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import numpy as np


class CenterAllFeaturesEstimator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.is_fitted_ = False
        self.column_means_ = None

    def fit(self, X, y=None):
        self.column_means_ = np.mean(X, axis=0)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise NotFittedError

        centered_X = X - self.column_means_
        return centered_X

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])