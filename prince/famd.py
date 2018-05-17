"""Factor Analysis of Mixed Data (FAMD)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils

from . import pca
from . import one_hot
from . import plot


class FAMD(pca.PCA):

    def __init__(self, n_components=2, n_iter=3, copy=True, random_state=None, engine='auto',
                 categorical_dtypes=('object', 'category')):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=copy,
            random_state=random_state,
            engine=engine
        )
        self.categorical_dtypes = categorical_dtypes

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate the categorical columns from the numerical ones
        cat = X.select_dtypes(include=self.categorical_dtypes)
        num = X.select_dtypes(exclude=self.categorical_dtypes)

        # One-hot encode the categorical columns
        self.one_hot_ = one_hot.OneHotEncoder().fit(cat)

        # Apply PCA to the indicator matrix
        return super().fit(pd.concat((num, self.one_hot_.transform(cat)), axis='columns'))

    def row_principal_coordinates(self, X):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self, 's_')

        return super().row_principal_coordinates(
            X=pd.concat(
                (
                    X.select_dtypes(exclude=self.categorical_dtypes),
                    self.one_hot_.transform(X.select_dtypes(include=self.categorical_dtypes))
                ),
                axis='columns'
            )
        )
