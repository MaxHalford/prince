"""Factor Analysis of Mixed Data (FAMD)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn import preprocessing

from prince import pca, utils


class FAMD(pca.PCA):
    """Factor Analysis of Mixed Data (FAMD).

    This class performs a Principal Component Analysis on a mixed
    dataset containing both quantitative and qualitative variables,
    following the framework of Pagès (FAMD).

    During the fitting:
        - Quantitative variables are standardized (mean=0, std=1).
        - Categorical variables are transformed using a one-hot encoding
            followed by a centering by category frequencies and scaling by sqrt(p),
            consistent with Multiple Correspondence Analysis (MCA).

    Row coordinates are obtained from the SVD decomposition.
    Supplementary variables are projected post fitting.
    Column coordinates, correlations and cosine similarities contain both active
    and supplementary variables (numerical and modalities representations).
    """

    def __init__(
        self,
        rescale_with_mean=False,
        rescale_with_std=False,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",
        categorical_columns=None,
    ) -> None:
        super().__init__(
            rescale_with_mean=rescale_with_mean,
            rescale_with_std=rescale_with_std,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        self.handle_unknown = handle_unknown
        categorical_columns = [] if categorical_columns is None else categorical_columns
        self.categorical_columns = categorical_columns

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

    @utils.check_is_dataframe_input
    def fit(self, X, y=None, supplementary_columns=None):
        """Fit a Factor Analysis of Mixed Data (FAMD) model.

        Quantitative variables are standardized, while qualitative variables
        are transformed through a disjunctive (one-hot) encoding followed by
        centering and scaling using category proportions, similarly to Multiple
        Correspondence Analysis (MCA).

        The resulting preprocessed matrix is then fitted using PCA.

        Arguments:
            X : pd.DataFrame
                Input data containing numerical and categorical variables.
            y : ignored
                Not used, present for sklearn compatibility.
            supplementary_columns : list of str, optional
                Columns treated as supplementary (not used in the construction
                of the factor space but projected afterward).

        Returns:
            self : FAMD
                Fitted model.
        """
        supplementary_columns = supplementary_columns or []

        # 1. Split active / supplementary columns
        active_columns = X.columns.difference(
            supplementary_columns,
            sort=False,
        ).tolist()

        self.active_columns_ = active_columns
        self.supplementary_columns_ = supplementary_columns

        # 2. Detect variable types

        # Active
        self.active_categorical_columns_ = [
            column for column in active_columns if column in self.categorical_columns
        ]
        if not self.active_categorical_columns_:
            raise ValueError("All variables are quantitative: PCA should be used")

        self.active_numerical_columns_ = [
            column for column in active_columns if column not in self.categorical_columns
        ]
        if not self.active_numerical_columns_:
            raise ValueError("All variables are qualitative: MCA should be used")

        # Supplemetary
        self.supplementary_categorical_columns_ = [
            column for column in supplementary_columns if column in self.categorical_columns
        ]

        self.supplementary_numerical_columns_ = [
            column for column in supplementary_columns if column not in self.categorical_columns
        ]

        # 3. Active numerical preprocessing
        Z_num_active = pd.DataFrame(index=X.index)

        if self.active_numerical_columns_:
            X_num_active = X[self.active_numerical_columns_].copy()

            self.active_numerical_scaler_ = preprocessing.StandardScaler(
                with_mean=True,
                with_std=True,
            )

            Z_num_active = pd.DataFrame(
                self.active_numerical_scaler_.fit_transform(X_num_active),
                columns=self.active_numerical_columns_,
                index=X.index,
            )

        # 4. Active categorical preprocessing
        X_cat_active = X[self.active_categorical_columns_].astype(str)
        G_active = pd.get_dummies(
            X_cat_active,
            dtype=float,
        )
        self.active_modalities_ = G_active.columns.tolist()
        self.active_categories_ = {col: X_cat_active[col].astype(str).unique() for col in self.active_categorical_columns_}

        p_active = G_active.mean(axis=0)
        self.active_modalities_proportions_ = p_active

        Z_cat_active = G_active.sub(p_active, axis=1).div(np.sqrt(p_active), axis=1)

        # 5. Supplementary numerical preprocessing
        Z_num_sup = pd.DataFrame(index=X.index)

        if self.supplementary_numerical_columns_:
            X_num_sup = X[self.supplementary_numerical_columns_].copy()

            self.supplementary_numerical_scaler_ = preprocessing.StandardScaler(
                with_mean=True,
                with_std=True,
            )

            Z_num_sup = pd.DataFrame(
                self.supplementary_numerical_scaler_.fit_transform(X_num_sup),
                columns=self.supplementary_numerical_columns_,
                index=X.index,
            )

        # 6. Supplementary categorical preprocessing
        Z_cat_sup = pd.DataFrame(index=X.index)

        self.supplementary_categorical_modalities_ = []

        if self.supplementary_categorical_columns_:
            X_cat_sup = X[self.supplementary_categorical_columns_].astype(str)
            G_sup = pd.get_dummies(
                X_cat_sup,
                dtype=float,
            )
            self.supplementary_categorical_modalities_ = G_sup.columns.tolist()

            p_sup = G_sup.mean(axis=0)
            self.supplementary_modalities_proportions_ = p_sup

            Z_cat_sup = G_sup.sub(p_sup, axis=1).div(np.sqrt(p_sup), axis=1)

        # 7. Build global preprocessed matrix
        Z = pd.concat(
            [
                Z_num_active,
                Z_cat_active,
                Z_num_sup,
                Z_cat_sup,
            ],
            axis=1,
        )

        self.preprocessed_column_names_ = Z.columns.tolist()

        # 8. Define supplementary columns inside Z
        supplementary_preprocessed_columns = list(Z_num_sup.columns) + list(Z_cat_sup.columns)

        # 9. PCA fit
        super().fit(
            Z,
            supplementary_columns=supplementary_preprocessed_columns,
        )

        # Determine column_coordinates_
        # This is based on line 184 in FactoMineR's famd.R file
        rc = self.row_coordinates(X)
        weights = np.ones(len(G_active)) / len(G_active)
        norm = (rc**2).multiply(weights, axis=0).sum()
        eta2_dict = {}
        for col in self.active_categorical_columns_:
            tt = G_active[[f"{col}_{c}" for c in self.active_categories_[col]]]
            ni = (tt / len(tt)).sum()
            eta2_dict[col] = (
                rc.apply(lambda x: (tt.multiply(x * weights, axis=0).sum() ** 2 / ni).sum()) / norm
            ).values
        eta2_active = pd.DataFrame(eta2_dict, index=rc.columns)
        # Save signed correlations for quantitative variables before squaring.
        # For standardized data, PCA column_coordinates_ = V.T * sqrt(eig) = correlations.
        # This corresponds to FactoMineR's quanti.var$coord.
        self._quanti_var_coord = self.column_coordinates_.loc[self.active_numerical_columns_].copy()
        self.active_column_coordinates_ = pd.concat([self._quanti_var_coord**2, eta2_active.T])
        self.active_column_coordinates_.columns.name = "component"
        self.active_column_coordinates_.index.name = "variable"

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        # Separate numerical columns from categorical columns
        X_num = X[self.active_categorical_columns_].copy()
        X_cat = X[self.active_categorical_columns_]

        # Preprocess numerical columns
        X_num[:] = self.active_numerical_scaler_.transform(X_num)

        # Preprocess categorical columns
        if self.handle_unknown == "error":
            for col in self.active_categorical_columns_:
                unknown = set(X_cat[col].astype(str).unique()) - set(self.active_categories_[col])
                if unknown:
                    raise ValueError(
                        f"Found unknown categories {unknown} in column '{col}' during transform."
                    )
        X_cat = pd.get_dummies(X_cat.astype(str), dtype=float).reindex(
            columns=self.supplementary_categorical_modalities_, fill_value=0
        )
        prop = X_cat.mean()
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

    @property
    @utils.check_is_fitted
    def column_correlations(self):
        """Correlations between variables and components.

        For quantitative variables, these are the signed Pearson correlations between
        each standardized variable and each principal component (FactoMineR: quanti.var$coord).
        These values form the correlation circle.

        For qualitative variables, signed correlations do not exist. The eta-squared (η²)
        values from column_coordinates_ are returned instead.
        """
        return pd.concat([self._quanti_var_coord, self.column_coordinates_.loc[self.active_categorical_columns_]])

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities_(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @property
    def column_contributions_(self):
        return self.column_coordinates_ / self.eigenvalues_
