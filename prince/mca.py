"""Multiple Correspondence Analysis (MCA)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.preprocessing
import sklearn.utils

from prince import utils

from . import ca


class MCA(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, ca.CA):
    def __init__(
        self,
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        one_hot=True,
        handle_unknown="error",
    ):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        self.one_hot = one_hot
        self.handle_unknown = handle_unknown

    def _prepare(self, X):
        if self.one_hot:
            # Create the one-hot encoder if it doesn't exist (usually because we're in the fit method)
            X = pd.get_dummies(X, columns=X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.arange(self.n_components_)

    @utils.check_is_dataframe_input
    def fit(self, X, y=None):
        """Fit the MCA for the dataframe X.

        The MCA is computed on the indicator matrix (i.e. `X.get_dummies()`). If some of the columns are already
        in indicator matrix format, you'll want to pass in `K` as the number of "real" variables that it represents.
        (That's used for correcting the inertia linked to each dimension.)

        """

        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, np.number])

        # K is the number of actual variables, to apply the Benzécri correction
        self.K_ = X.shape[1]

        # One-hot encode the data
        one_hot = self._prepare(X)

        # We need the number of columns to apply the Greenacre correction
        self.J_ = one_hot.shape[1]

        # Apply CA to the indicator matrix
        super().fit(one_hot)

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        return super().row_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        oh = self._prepare(X)
        return super()._row_cosine_similarities(X=oh, F=super().row_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_coordinates(self, X):
        return super().column_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities(self, X):
        oh = self._prepare(X)
        return super()._column_cosine_similarities(X=oh, G=super().column_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)
