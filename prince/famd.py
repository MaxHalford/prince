"""Factor Analysis of Mixed Data (FAMD)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn import preprocessing

from prince import pca, utils


class FAMD(pca.PCA):
    def __init__(
        self,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",
    ):
        super().__init__(
            rescale_with_mean=True,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        self.handle_unknown = handle_unknown

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

    @utils.check_is_dataframe_input
    def fit(self, X, y=None):
        # Separate numerical columns from categorical columns
        self.num_cols_ = X.select_dtypes(include=["float"]).columns.tolist()
        if not self.num_cols_:
            raise ValueError("All variables are qualitative: MCA should be used")
        self.cat_cols_ = X.columns.difference(self.num_cols_).tolist()
        if not self.cat_cols_:
            raise ValueError("All variables are quantitative: PCA should be used")

        # Preprocess numerical columns
        X_num = X[self.num_cols_].copy()
        self.num_scaler_ = preprocessing.StandardScaler().fit(X_num)
        X_num[:] = self.num_scaler_.transform(X_num)

        # Preprocess categorical columns
        X_cat = X[self.cat_cols_]
        X_cat_oh = pd.get_dummies(X_cat.astype(str), dtype=float)
        self.dummy_columns_ = X_cat_oh.columns.tolist()
        self.categories_ = {col: X_cat[col].astype(str).unique() for col in self.cat_cols_}
        prop = X_cat_oh.sum() / X_cat_oh.sum().sum() * 2
        X_cat_oh_norm = X_cat_oh.sub(X_cat_oh.mean(axis="rows")).div(prop**0.5, axis="columns")

        Z = pd.concat([X_num, X_cat_oh_norm], axis=1)
        super().fit(Z)

        # Determine column_coordinates_
        # This is based on line 184 in FactoMineR's famd.R file
        rc = self.row_coordinates(X)
        weights = np.ones(len(X_cat_oh)) / len(X_cat_oh)
        norm = (rc**2).multiply(weights, axis=0).sum()
        eta2 = pd.DataFrame(index=rc.columns)
        for col in self.cat_cols_:
            tt = X_cat_oh[[f"{col}_{c}" for c in self.categories_[col]]]
            ni = (tt / len(tt)).sum()
            eta2[col] = (
                rc.apply(lambda x: (tt.multiply(x * weights, axis=0).sum() ** 2 / ni).sum()) / norm
            ).values
        self.column_coordinates_ = pd.concat(
            [self.column_coordinates_.loc[self.num_cols_] ** 2, eta2.T]
        )
        self.column_coordinates_.columns.name = "component"
        self.column_coordinates_.index.name = "variable"

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        # Separate numerical columns from categorical columns
        X_num = X[self.num_cols_].copy()
        X_cat = X[self.cat_cols_]

        # Preprocess numerical columns
        X_num[:] = self.num_scaler_.transform(X_num)

        # Preprocess categorical columns
        if self.handle_unknown == "error":
            for col in self.cat_cols_:
                unknown = set(X_cat[col].astype(str).unique()) - set(self.categories_[col])
                if unknown:
                    raise ValueError(
                        f"Found unknown categories {unknown} in column '{col}' during transform."
                    )
        X_cat = pd.get_dummies(X_cat.astype(str), dtype=float).reindex(
            columns=self.dummy_columns_, fill_value=0
        )
        prop = X_cat.sum() / X_cat.sum().sum() * 2
        X_cat = X_cat.sub(X_cat.mean(axis="rows")).div(prop**0.5, axis="columns")

        Z = pd.concat([X_num, X_cat], axis=1).fillna(0.0)

        return super().row_coordinates(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def inverse_transform(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_standard_coordinates(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_correlations(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities_(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @property
    def column_contributions_(self):
        return self.column_coordinates_ / self.eigenvalues_
