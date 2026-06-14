"""Multiple Correspondence Analysis (MCA)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.preprocessing
import sklearn.utils

from prince import utils

from . import ca


class MCA(ca.CA, sklearn.base.TransformerMixin):
    """Multiple Correspondence Analysis (MCA).

    MCA extends correspondence analysis to more than two categorical variables. It works by
    one-hot encoding the input, then applying CA to the resulting indicator matrix. This
    produces a low-dimensional representation of the associations between categories.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to compute.
    n_iter : int, default=10
        Number of iterations for the SVD solver.
    copy : bool, default=True
        Whether to copy the input data before fitting.
    check_input : bool, default=True
        Whether to validate the input array.
    random_state : int or None, default=None
        Seed for the random number generator (used in the SVD solver).
    engine : str, default='sklearn'
        SVD engine to use. Either 'sklearn' or 'scipy'.
    one_hot : bool, default=True
        Whether to one-hot encode the input. Set to False if the data is already
        in indicator matrix format.
    one_hot_prefix_sep : str, default='__'
        Separator used between the column name and the category when one-hot encoding.
    one_hot_columns_to_drop : list of str or None, default=None
        Columns to drop after one-hot encoding (useful for removing redundant indicators).
    correction : str or None, default=None
        Eigenvalue correction method. Either 'benzecri', 'greenacre', or None.
        Can only be used when ``one_hot=True``.

    """

    def __init__(
        self,
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        one_hot=True,
        one_hot_prefix_sep="__",
        one_hot_columns_to_drop=None,
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
        self.one_hot_prefix_sep = one_hot_prefix_sep
        self.one_hot_columns_to_drop = one_hot_columns_to_drop
        self.correction = correction

    def _one_hot(self, X):
        if self.one_hot:
            return pd.get_dummies(X, columns=X.columns, prefix_sep=self.one_hot_prefix_sep)
        return X

    def _prepare(self, X):
        """One-hot encode the input if needed, and align columns with the fitted indicator matrix."""
        X = self._one_hot(X)
        if self.one_hot and self.one_hot_columns_to_drop is not None:
            X = X.drop(columns=self.one_hot_columns_to_drop, errors="ignore")
        if (one_hot_columns_ := getattr(self, "one_hot_columns_", None)) is not None:
            X = X.reindex(columns=one_hot_columns_.union(X.columns), fill_value=False)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.arange(self.n_components_)

    def _subset_greenacre_quantities(self):
        """Adjusted eigenvalues and total inertia for subset MCA with Greenacre correction.

        Ports the ``lambda = "adjusted"`` + ``subsetcat`` branch of R's ``ca::mjca``
        (Nenadić & Greenacre 2007; *Correspondence Analysis in Practice* ch. 19 & 21).
        Notation follows the R source so the two implementations can be cross-checked:

        - ``B``    : Burt matrix of the full (pre-drop) indicator matrix, ``Zᵀ Z``.
        - ``P``    : ``B / sum(B)`` — Burt as a joint probability table.
        - ``cm``   : column marginals of ``P`` — the standard MCA column masses.
                     Computed on the *full* Burt so dropping categories does not
                     redistribute mass (this is what makes the correction "subset"-aware).
        - ``B_null``, ``S_null`` : Burt with self-counts (the literal diagonal) zeroed
                     and standardised. For an indicator-derived Burt, the literal
                     diagonal *is* each variable's block-diagonal, so ``B_null`` is the
                     off-block-diagonal Burt — the part that carries inter-variable
                     association. Its eigendecomposition on the active subset yields
                     the principal inertias.
        - ``Pe``, ``Se`` : ``P`` with each variable's block-diagonal replaced by the
                     marginal product ``cmᵇ cmᵇᵀ`` (independence within each variable).
                     ``sum(Se² ) · Q/(Q-1)`` on the active subset is the total adjusted
                     inertia — what the per-dimension inertias are expressed against.
        - ``Q``    : number of original variables (``self.K_``).
        - ``mask`` : boolean selector of active one-hot columns (post-drop).
        """
        B = self._subset_full_burt_
        cm = B.sum(axis=0) / B.sum()
        sqrt_cm_outer = np.sqrt(np.outer(cm, cm))
        cm_outer = np.outer(cm, cm)

        # S_null: standardised residuals of the off-block-diagonal Burt.
        B_null = B - np.diag(np.diag(B))
        S_null = (B_null / B_null.sum() - cm_outer) / sqrt_cm_outer

        # Se: standardised residuals under the "independence within each variable" model.
        Pe = (B / B.sum()).copy()
        for q in range(self.K_):
            idx = np.where(self._subset_col_to_var_ == q)[0]
            Pe[np.ix_(idx, idx)] = np.outer(cm[idx], cm[idx])
        Se = (Pe - cm_outer) / sqrt_cm_outer

        # Principal inertias on the active subset; clip numerical negatives.
        mask = self._subset_mask_
        eigvals = np.linalg.eigvalsh(S_null[np.ix_(mask, mask)])[::-1]
        lambda_adj = np.clip(eigvals, 0, None) ** 2
        # Total adjusted inertia. The Q/(Q-1) factor mirrors the non-subset Greenacre
        # adjustment (see eqn. 19.5 in Greenacre, CA in Practice).
        Q = self.K_
        lambda_t = (Se[np.ix_(mask, mask)] ** 2).sum() * Q / (Q - 1)
        return lambda_adj, lambda_t

    @property
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        eigenvalues = super().eigenvalues_
        # Benzécri and Greenacre corrections
        if self.correction in {"benzecri", "greenacre"}:
            if self.correction == "greenacre" and self.one_hot_columns_to_drop is not None:
                lambda_adj, _ = self._subset_greenacre_quantities()
                return lambda_adj[: len(eigenvalues)]
            K = self.K_
            return np.array(
                [(K / (K - 1) * (eig - 1 / K)) ** 2 if eig > 1 / K else 0 for eig in eigenvalues]
            )
        return eigenvalues

    @property
    @utils.check_is_fitted
    def percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        # Greenacre correction on a subset MCA: closed-form Benzécri assumes uniform row
        # sums in the indicator matrix and so mis-calibrates when categories are dropped.
        if self.correction == "greenacre" and self.one_hot_columns_to_drop is not None:
            lambda_adj, lambda_t = self._subset_greenacre_quantities()
            n = len(super().eigenvalues_)
            return 100 * lambda_adj[:n] / lambda_t
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
        """Fit the MCA on a categorical dataframe.

        The input is one-hot encoded into an indicator matrix (unless ``one_hot=False``),
        then correspondence analysis is applied. The number of original variables in ``X``
        is stored as ``K_`` and used for the Benzécri/Greenacre eigenvalue corrections.

        Parameters
        ----------
        X : DataFrame
            A dataframe where each column is a categorical variable.
        y : ignored

        Returns
        -------
        self

        """

        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

        # K is the number of actual variables, to apply the Benzécri correction
        self.K_ = X.shape[1]

        # One-hot encode the data. The full (pre-drop) indicator is kept so the
        # subset-Greenacre correction can recover the full Burt marginals.
        full_one_hot = self._one_hot(X)
        one_hot = full_one_hot
        if self.one_hot and self.one_hot_columns_to_drop is not None:
            one_hot = one_hot.drop(columns=self.one_hot_columns_to_drop, errors="ignore")
        self.one_hot_columns_ = one_hot.columns

        # We need the number of columns to apply the Greenacre correction
        self.J_ = one_hot.shape[1]

        if (
            self.one_hot
            and self.one_hot_columns_to_drop is not None
            and self.correction == "greenacre"
        ):
            Z_full = full_one_hot.to_numpy(dtype=float)
            self._subset_full_burt_ = Z_full.T @ Z_full
            var_index = {col: i for i, col in enumerate(X.columns)}
            sep = self.one_hot_prefix_sep
            self._subset_col_to_var_ = np.array(
                [var_index[c.rsplit(sep, 1)[0]] for c in full_one_hot.columns]
            )
            self._subset_mask_ = full_one_hot.columns.isin(one_hot.columns).astype(bool)

        # Apply CA to the indicator matrix
        super().fit(one_hot)

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        """Row principal coordinates in the MCA space."""
        return super().row_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        """Squared cosine similarities (cos2) of each row on each component."""
        oh = self._prepare(X)
        return super()._row_cosine_similarities(X=oh, F=super().row_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_coordinates(self, X):
        """Column (category) principal coordinates in the MCA space."""
        return super().column_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities(self, X):
        """Squared cosine similarities (cos2) of each column on each component."""
        oh = self._prepare(X)
        return super()._column_cosine_similarities(X=oh, G=super().column_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])
        return self.row_coordinates(X)
