"""Factor Analysis of Mixed Data (FAMD)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn import preprocessing

from prince import pca, utils


class FAMD(pca.PCA):
    """Factor Analysis of Mixed Data (FAMD).

    FAMD applies a PCA to a mixed-type table: quantitative variables are standardized,
    qualitative variables are one-hot encoded and then centered and rescaled by
    ``1/sqrt(p_j)`` where ``p_j`` is the proportion of each modality. This balances
    the influence of numerical and categorical variables.

    Parameters
    ----------
    categorical_columns : list of str, optional
        Columns to treat as categorical. If ``None``, columns are auto-detected
        based on dtype (non-float columns are treated as categorical).

    Attributes
    ----------
    num_cols_ : list of str
        All numerical columns seen at fit time (active + supplementary).
    cat_cols_ : list of str
        All categorical columns seen at fit time (active + supplementary).
    num_scaler_ : sklearn.preprocessing.StandardScaler
        Fitted on the active numerical columns. ``num_scaler_.feature_names_in_``
        is the canonical list of active numerical columns; supplementary numerical
        columns are ``set(num_cols_) - set(num_scaler_.feature_names_in_)``.
    one_hot_columns_ : list of str
        One-hot encoded modality names for the active categoricals.
    categories_ : dict[str, ndarray]
        Maps each active categorical column to its observed categories. The set of
        keys is the list of active categorical columns; supplementary categoricals
        are ``set(cat_cols_) - set(categories_)``.
    """

    def __init__(
        self,
        rescale_with_mean=True,
        rescale_with_std=False,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",
        categorical_columns=None,
    ):
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
        self.categorical_columns = categorical_columns

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

    def _split_column_types(self, X):
        if self.categorical_columns is None:
            num_cols = X.select_dtypes(include=["float"]).columns.tolist()
            cat_cols = X.columns.difference(num_cols, sort=False).tolist()
        else:
            cat_set = set(self.categorical_columns)
            cat_cols = [c for c in X.columns if c in cat_set]
            num_cols = [c for c in X.columns if c not in cat_set]
        return num_cols, cat_cols

    def _modalities_for(self, col):
        """Modality column names for an active categorical, in training order."""
        return [f"{col}_{cat}" for cat in self.categories_[col]]

    def _active_num_cols(self):
        return list(self.num_scaler_.feature_names_in_)

    def _active_cat_cols(self):
        return list(self.categories_)

    @utils.check_is_dataframe_input
    def fit(self, X, y=None, supplementary_columns=None):
        """Fit a Factor Analysis of Mixed Data (FAMD) model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with both numerical and categorical columns.
        supplementary_columns : list of str, optional
            Columns to treat as supplementary (projected onto the factor space
            after fitting, not used to compute it).
        """
        supplementary_columns = list(supplementary_columns or [])

        self.num_cols_, self.cat_cols_ = self._split_column_types(X)
        if not self.num_cols_:
            raise ValueError("All variables are qualitative: MCA should be used")
        if not self.cat_cols_:
            raise ValueError("All variables are quantitative: PCA should be used")

        sup_set = set(supplementary_columns)
        active_num = [c for c in self.num_cols_ if c not in sup_set]
        active_cat = [c for c in self.cat_cols_ if c not in sup_set]
        sup_num = [c for c in self.num_cols_ if c in sup_set]
        sup_cat = [c for c in self.cat_cols_ if c in sup_set]

        if not active_num:
            raise ValueError("No active numerical variables left after removing supplementary ones")
        if not active_cat:
            raise ValueError(
                "No active categorical variables left after removing supplementary ones"
            )

        # Cast to float so int columns can be passed as numerical via
        # `categorical_columns=[...]` — `X_num[:] = ...` refuses to assign float
        # arrays into int dtypes.
        X_num = X[active_num].astype(float)
        self.num_scaler_ = preprocessing.StandardScaler().fit(X_num)
        X_num[:] = self.num_scaler_.transform(X_num)

        X_cat_oh, X_cat_oh_norm = _mca_code(X[active_cat])
        self.one_hot_columns_ = X_cat_oh.columns.tolist()
        self.categories_ = {col: X[col].astype(str).unique() for col in active_cat}

        Z_parts = [X_num, X_cat_oh_norm]
        sup_preprocessed_columns = []
        if sup_num:
            X_sup_num = X[sup_num].astype(float)
            X_sup_num[:] = preprocessing.StandardScaler().fit_transform(X_sup_num)
            Z_parts.append(X_sup_num)
            sup_preprocessed_columns.extend(sup_num)
        if sup_cat:
            X_sup_oh, X_sup_oh_norm = _mca_code(X[sup_cat])
            Z_parts.append(X_sup_oh_norm)
            sup_preprocessed_columns.extend(X_sup_oh.columns.tolist())

        Z = pd.concat(Z_parts, axis=1)
        super().fit(Z, supplementary_columns=sup_preprocessed_columns)
        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        active_num = self._active_num_cols()
        active_cat = self._active_cat_cols()

        X_num = X[active_num].astype(float)
        X_num[:] = self.num_scaler_.transform(X_num)

        X_cat = X[active_cat]
        if self.handle_unknown == "error":
            for col in active_cat:
                unknown = set(X_cat[col].astype(str).unique()) - set(self.categories_[col])
                if unknown:
                    raise ValueError(
                        f"Found unknown categories {unknown} in column '{col}' during transform."
                    )
        # Recompute proportions from the input (matches the historical FAMD
        # transform behavior — kept for backward compatibility with #169 doctest).
        X_cat_oh = pd.get_dummies(X_cat.astype(str), dtype=float).reindex(
            columns=self.one_hot_columns_, fill_value=0
        )
        p = X_cat_oh.mean(axis="rows")
        X_cat_norm = X_cat_oh.sub(p, axis="columns").div(np.sqrt(p), axis="columns")

        Z = pd.concat([X_num, X_cat_norm], axis=1).fillna(0.0)
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
        """Signed Pearson correlations for numerical variables, η² for categoricals.

        For quantitative variables: signed correlations with each principal component
        (FactoMineR: ``quanti.var$coord``) — the values plotted in the correlation circle.

        For qualitative variables: η²(q, s) = Σ_{k_q} G_s(k_q)² where G_s(k_q) is the
        PCA coordinate of modality k_q (Pagès 2004, §5.1 p. 98–99). The cross-factors
        between G_s and the barycentric form F_s cancel because of the FAMD modality
        scaling, leaving the sum of squared modality coordinates.
        """
        num_corr = self.column_coordinates_.loc[self._active_num_cols()]
        eta2_rows = {
            col: (self.column_coordinates_.loc[self._modalities_for(col)].to_numpy() ** 2).sum(
                axis=0
            )
            for col in self._active_cat_cols()
        }
        eta2 = pd.DataFrame(eta2_rows, index=self.column_coordinates_.columns).T
        return pd.concat([num_corr, eta2])

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities_(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @property
    def column_contributions_(self):
        """Contribution of each preprocessed column (modality or numerical) to each
        component: ``f² / λ`` where ``f`` is the PCA coordinate and ``λ`` the eigenvalue.
        """
        return self.column_coordinates_**2 / self.eigenvalues_


def _mca_code(X_cat):
    """One-hot encode and apply Pagès's MCA coding ``(X_oh - p) / sqrt(p)``.

    Returns the raw one-hot matrix (for column names) and the centered/scaled one.
    """
    X_oh = pd.get_dummies(X_cat.astype(str), dtype=float)
    p = X_oh.mean(axis="rows")
    X_oh_scaled = X_oh.sub(p, axis="columns").div(np.sqrt(p), axis="columns")
    return X_oh, X_oh_scaled
