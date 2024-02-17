import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column=None):
        self.target_column = target_column
        self.is_fitted_ = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        assert y is None

        # IMPLEMENT ME

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise NotFittedError

        # IMPLEMENT ME

        return X
