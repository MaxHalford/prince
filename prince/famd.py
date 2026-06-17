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

    def _preprocess_categorical(self, X_cat):
        """One-hot encode and apply Pagès's MCA coding (X_oh - p) / sqrt(p)."""
        X_oh = pd.get_dummies(X_cat.astype(str), dtype=float)
        p = X_oh.mean(axis="rows")
        X_oh_scaled = X_oh.sub(p, axis="columns").div(np.sqrt(p), axis="columns")
        return X_oh, X_oh_scaled

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

        # Detect column types across the full input (active + supplementary).
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

        # Active numerical: standardize. ``num_scaler_.feature_names_in_`` is the
        # canonical record of which numerical columns are active; supplementary
        # numerical columns are not surfaced as their own attribute and can be
        # recovered as ``set(num_cols_) - set(num_scaler_.feature_names_in_)``.
        # Cast to float upfront so integer columns can be passed as numerical via
        # ``categorical_columns=[...]`` — pandas refuses to assign float arrays into
        # int dtypes via ``X_num[:] = ...``.
        X_num = X[active_num].astype(float)
        self.num_scaler_ = preprocessing.StandardScaler().fit(X_num)
        X_num[:] = self.num_scaler_.transform(X_num)

        # Active categorical: one-hot encode + center + scale by sqrt(p). Similarly,
        # ``one_hot_columns_`` is the active-modality record; supplementary categoricals
        # can be recovered as ``[c for c in cat_cols_ if not any one_hot_columns_ starts
        # with "c_"]``.
        X_cat_oh, X_cat_oh_norm = self._preprocess_categorical(X[active_cat])
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
            X_sup_oh, X_sup_oh_norm = self._preprocess_categorical(X[sup_cat])
            Z_parts.append(X_sup_oh_norm)
            sup_preprocessed_columns.extend(X_sup_oh.columns.tolist())

        Z = pd.concat(Z_parts, axis=1)
        super().fit(Z, supplementary_columns=sup_preprocessed_columns)

        return self

    def _active_num_cols(self):
        return list(self.num_scaler_.feature_names_in_)

    def _active_cat_cols(self):
        # A categorical is active iff at least one of its modalities is in one_hot_columns_.
        return [
            c
            for c in self.cat_cols_
            if any(m.startswith(f"{c}_") for m in self.one_hot_columns_)
        ]

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
        X_cat = pd.get_dummies(X_cat.astype(str), dtype=float).reindex(
            columns=self.one_hot_columns_, fill_value=0
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
        """Signed correlations between numerical variables and components.

        For quantitative variables, these are the signed Pearson correlations
        (FactoMineR: ``quanti.var$coord``). They populate the correlation circle.
        Categorical variables are returned as η² values (no signed correlation
        exists for them).
        """
        active_num = self._active_num_cols()
        active_cat = self._active_cat_cols()

        num_corr = self.column_coordinates_.loc[active_num]

        # η²(q, s) = Σ_{k_q} G_s(k_q)² where G_s(k_q) is the PCA coord on the MCA-coded
        # indicator (Pagès 2004, §5.1 p.98–99). The barycentre form F_s(k_q) and G_s(k_q)
        # are related by G_s = √(p/λ)·F_s, and substituting into the variance-ratio
        # definition of η² makes the p_j and λ factors cancel, giving Σ G_s².
        eta2_rows = {}
        for col in active_cat:
            mods = [m for m in self.one_hot_columns_ if m.startswith(f"{col}_")]
            f = self.column_coordinates_.loc[mods].to_numpy()
            eta2_rows[col] = (f**2).sum(axis=0)
        eta2 = pd.DataFrame(eta2_rows, index=self.column_coordinates_.columns).T

        return pd.concat([num_corr, eta2])

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities_(self, X):
        raise NotImplementedError("FAMD inherits from PCA, but this method is not implemented yet")

    @property
    def column_contributions_(self):
        """Contribution of each (modality or numerical) variable to each component.

        For genuine PCA coordinates ``f``, the contribution is ``f² / λ`` (where ``λ`` is
        the eigenvalue). Rows are columns of the preprocessed indicator/numerical matrix.
        """
        return self.column_coordinates_**2 / self.eigenvalues_
