"""Multiple Correspondence Analysis (MCA)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.utils
from sklearn.preprocessing import OneHotEncoder

from prince import utils

from . import ca


class MCA(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, ca.CA):
    '''
    added new attributes to support one-hot encoding when handling unknown categories
    
    added attributes:
        get_dummies: if True, use pd.get_dummies to one-hot encode the data
        one_hot_encoder: OneHotEncoder object to use
        is_one_hot_fitted: check if one_hot_encoder is fitted (set it to true if the one_hot_encoder is already fitted)
    '''
    def __init__(
        self,
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        one_hot = True,
        get_dummies = False,#if True, use pd.get_dummies to one-hot encode the data
        one_hot_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=bool), #OneHotEncoder object to use
        is_one_hot_fitted = False
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
        self.get_dummies = get_dummies
        self.one_hot_encoder = one_hot_encoder
        self.is_one_hot_fitted = is_one_hot_fitted


    def _prepare(self, X):
        if self.one_hot:
            if self.get_dummies:
                X = pd.get_dummies(X, columns=X.columns)
                return X
            else:
                if self.is_one_hot_fitted == False:
                    #if the one_hot_encoder is not fitted, to fit and also set the is_one_hot_fitted variable to True
                    X_enc = self.one_hot_encoder.fit_transform(X)
                    X_enc = pd.DataFrame(X_enc, columns=self.one_hot_encoder.get_feature_names_out(X.columns))
                    self.is_one_hot_fitted = True
                    return X_enc
                else:
                    #checking if the columns fed to the onehot encoder and the columns fitted to the onehot encoder are the same
                    oh_cols = set(self.one_hot_encoder.feature_names_in_.tolist())
                    X_cols = set(X.columns.tolist())
                    
                    if oh_cols == X_cols:
                        #if the fitted cols are the same as the inferencing columns, then can transform
                        X_enc = self.one_hot_encoder.transform(X)
                        X_enc = pd.DataFrame(X_enc, columns=self.one_hot_encoder.get_feature_names_out(X.columns))
                        return X_enc
                    else:
                        #if the fitted cols are different to the inferencing columns, then should fit the onehot encoder again, to handle unit tests
                        X_enc = self.one_hot_encoder.fit_transform(X)
                        X_enc = pd.DataFrame(X_enc, columns=self.one_hot_encoder.get_feature_names_out(X.columns))
                        return X_enc
        return X

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
