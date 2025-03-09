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
        correction=None,
    ):
        if correction is not None:
            if correction not in {"benzecri", "greenacre"}:
                raise ValueError("correction must be either 'benzecri' or 'greenacre' if provided.")
            if not one_hot:
                raise ValueError(
                    "correction can only be applied when one_hot is True. This is because the "
                    "number of original variables is needed to apply the correction."
                )

        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        self.one_hot = one_hot
        self.correction = correction

    def _prepare(self, X):
        if self.one_hot:
            X = pd.get_dummies(X, columns=X.columns, prefix_sep="__")
            if (one_hot_columns_ := getattr(self, "one_hot_columns_", None)) is not None:
                X = X.reindex(columns=one_hot_columns_.union(X.columns), fill_value=False)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.arange(self.n_components_)

    @property
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        eigenvalues = super().eigenvalues_
        # Benzécri and Greenacre corrections
        if self.correction in {"benzecri", "greenacre"}:
            K = self.K_
            return np.array(
                [(K / (K - 1) * (eig - 1 / K)) ** 2 if eig > 1 / K else 0 for eig in eigenvalues]
            )
        return eigenvalues

    @property
    @utils.check_is_fitted
    def percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        # Benzécri correction
        if self.correction == "benzecri":
            eigenvalues = self.eigenvalues_
            return 100 * eigenvalues / eigenvalues.sum()
        # Greenacre correction
        if self.correction == "greenacre":
            eigenvalues = super().eigenvalues_
            benzecris = self.eigenvalues_
            K, J = (self.K_, self.J_)
            average_inertia = (K / (K - 1)) * ((eigenvalues**2).sum() - (J - K) / K**2)
            return 100 * benzecris / average_inertia
        # No correction
        return super().percentage_of_variance_

    @utils.check_is_dataframe_input
    def fit(self, X, y=None):
        """Fit the MCA for the dataframe X.

        The MCA is computed on the indicator matrix (i.e. `X.get_dummies()`). If some of the columns are already
        in indicator matrix format, you'll want to pass in `K` as the number of "real" variables that it represents.
        (That's used for correcting the inertia linked to each dimension.)

        """

        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

        # K is the number of actual variables, to apply the Benzécri correction
        self.K_ = X.shape[1]

        # One-hot encode the data
        one_hot = self._prepare(X)
        self.one_hot_columns_ = one_hot.columns

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
            sklearn.utils.check_array(X, dtype=[str, "numeric"])
        return self.row_coordinates(X)
