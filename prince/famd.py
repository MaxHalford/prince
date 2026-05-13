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
        rescale_with_mean=False,
        rescale_with_std=False,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        handle_unknown="error",
        numerical_columns=None,
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

        numerical_columns = numerical_columns or []
        categorical_columns = categorical_columns or []

        overlap = set(numerical_columns) & set(categorical_columns)

        if overlap:
            raise ValueError(f"Columns cannot be both numerical and categorical: {overlap}")

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

    @utils.check_is_dataframe_input
    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        supplementary_columns=None,
    ):
        """Fit a Factor Analysis of Mixed Data (FAMD) model.

        This method performs a Principal Component Analysis on a mixed
        dataset containing both quantitative and qualitative variables,
        following the framework of Pagès (Multiple Factor Analysis / FAMD).

        Quantitative variables are standardized, while qualitative variables
        are transformed through a disjunctive (one-hot) encoding followed by
        centering and scaling using category proportions, similarly to Multiple
        Correspondence Analysis (MCA).


        The resulting preprocessed matrix is then analyzed using PCA.

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

        Notes:
            - Quantitative variables are standardized (mean=0, std=1).
            - Categorical variables are transformed using a one-hot encoding
                followed by a centering by category frequencies and scaling by sqrt(p),
                consistent with Multiple Correspondence Analysis (MCA).
            - PCA is performed on the concatenated preprocessed matrix.
            - Row coordinates are obtained from the SVD decomposition.
            - Supplementary variables are projected post-hoc
            - Column coordinates contain both active and supplementary variables
                (numerical and categorical representations).
        """
        supplementary_columns = supplementary_columns or []

        # ============================================================
        # 1. Split active / supplementary variables
        # ============================================================

        active_columns = X.columns.difference(
            supplementary_columns,
            sort=False,
        ).tolist()

        self.feature_names_in_ = active_columns
        self.supplementary_columns_ = supplementary_columns

        X_active = X[active_columns].copy()

        # ============================================================
        # 2. Detect variable types
        # ============================================================

        user_num_active = [col for col in active_columns if col in self.numerical_columns]
        user_cat_active = [col for col in active_columns if col in self.categorical_columns]

        remaining_active = X_active.columns.difference(
            user_num_active + user_cat_active,
            sort=False,
        )

        auto_num_active = (
            X_active[remaining_active].select_dtypes(include=["number"]).columns.tolist()
        )

        auto_cat_active = remaining_active.difference(auto_num_active).tolist()

        self.num_cols_active_ = user_num_active + auto_num_active
        self.cat_cols_active_ = user_cat_active + auto_cat_active

        # ============================================================
        # 3. Numerical preprocessing
        # ============================================================

        X_num_active = X_active[self.num_cols_active_].copy()

        self.num_scaler_active_ = preprocessing.StandardScaler(
            with_mean=True,
            with_std=True,
        )

        Z_num_active = pd.DataFrame(
            self.num_scaler_active_.fit_transform(X_num_active),
            columns=self.num_cols_active_,
            index=X.index,
        )

        # ============================================================
        # 4. Categorical preprocessing
        # ============================================================

        X_cat_active = X_active[self.cat_cols_active_].astype(str)

        G_active = pd.get_dummies(
            X_cat_active,
            dtype=float,
        )

        self.active_modalities_ = G_active.columns.tolist()

        p = G_active.mean(axis=0)
        self.active_modality_proportions_ = p

        Z_cat_active = G_active.sub(
            p,
            axis=1,
        ).div(
            np.sqrt(p),
            axis=1,
        )

        # ============================================================
        # 5. Build global matrix
        # ============================================================

        Z = pd.concat(
            [Z_num_active, Z_cat_active],
            axis=1,
        )

        self._Z_active_columns_ = Z.columns.tolist()

        # ============================================================
        # 6. Fit PCA on already preprocessed matrix
        # ============================================================

        super().fit(Z)

        # ============================================================
        # 7. Row coordinates
        # ============================================================

        F = pd.DataFrame(
            self.svd_.U * self.eigenvalues_**0.5,
            index=X.index,
        )

        F.columns.name = "component"

        self.row_coordinates_ = F

        # ============================================================
        # 8. Numerical variable coordinates
        # correlations/loadings
        # ============================================================

        self.numerical_coordinates_active_ = self.column_coordinates_.loc[self.num_cols_active_]

        self.numerical_coordinates_active_.index.name = "variable"
        self.numerical_coordinates_active_.columns.name = "component"

        # ============================================================
        # 9. Modalities coordinates
        # barycenters of individuals
        # ============================================================

        self.modality_coordinates_active_ = (G_active.T @ F).div(G_active.sum(axis=0), axis=0)

        self.modality_coordinates_active_.index.name = "modality"
        self.modality_coordinates_active_.columns.name = "component"

        # ============================================================
        # 10. Supplementary variables
        # ============================================================

        self.supplementary_numerical_columns_ = []
        self.supplementary_categorical_columns_ = []

        self.supplementary_numerical_coordinates_ = pd.DataFrame()
        self.supplementary_modality_coordinates_ = pd.DataFrame()

        if supplementary_columns:
            X_sup = X[supplementary_columns].copy()

            num_sup = [col for col in supplementary_columns if col in self.numerical_columns]
            cat_sup = [col for col in supplementary_columns if col in self.categorical_columns]

            remaining_sup = [col for col in supplementary_columns if col not in num_sup + cat_sup]

            auto_num_sup = X_sup[remaining_sup].select_dtypes(include=["number"]).columns.tolist()
            auto_cat_sup = [col for col in remaining_sup if col not in auto_num_sup]

            self.num_cols_sup_ = num_sup + auto_num_sup
            self.cat_cols_sup_ = cat_sup + auto_cat_sup

            # --------------------------------------------------------
            # supplementary quantitative
            # projected by correlations
            # --------------------------------------------------------

            if self.num_cols_sup_:
                X_num_sup = X_sup[self.num_cols_sup_]

                self.num_scaler_sup_ = preprocessing.StandardScaler(
                    with_mean=True,
                    with_std=True,
                )

                Z_num_sup = pd.DataFrame(
                    self.num_scaler_sup_.fit_transform(X_num_sup),
                    columns=self.num_cols_sup_,
                    index=X.index,
                )

                self.numerical_coordinates_sup_ = (Z_num_sup.T @ self.row_coordinates_).div(
                    np.sqrt(self.eigenvalues_), axis=1
                ) / len(X)

                self.numerical_coordinates_sup_.index.name = "variable"
                self.numerical_coordinates_sup_.columns.name = "component"

            # --------------------------------------------------------
            # supplementary categorical
            # projected as barycenters
            # --------------------------------------------------------

            if self.cat_cols_sup_:
                X_cat_sup = X_sup[self.cat_cols_sup_].astype(str)

                G_sup = pd.get_dummies(
                    X_cat_sup,
                    dtype=float,
                )

                self.modality_coordinates_sup_ = (G_sup.T @ F).div(G_sup.sum(axis=0), axis=0)

                self.modality_coordinates_sup_.index.name = "modality"
                self.modality_coordinates_sup_.columns.name = "component"

        # 11. Final coordinates (with sup)
        coords_to_concat = [
            self.numerical_coordinates_active_,
            self.modality_coordinates_active_,
        ]

        if self.num_cols_sup_:
            coords_to_concat.append(self.numerical_coordinates_sup_)

        if self.cat_cols_sup_:
            coords_to_concat.append(self.modality_coordinates_sup_)

        self.column_coordinates_ = pd.concat(coords_to_concat)

        self.column_coordinates_.columns.name = "component"

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        """
        Return the principal coordinates of the rows.

        Only active variables are used to reconstruct the FAMD space.
        Supplementary variables do not participate in the projection:
        they are projected separately after the factor space is built.
        """

        # ============================================================
        # 1. ACTIVE NUMERICAL VARIABLES
        # ============================================================
        X_num_active = X[self.num_cols_active_].copy()

        Z_num_active = pd.DataFrame(
            self.num_scaler_active_.transform(X_num_active),
            index=X.index,
            columns=self.num_cols_active_,
        )

        # ============================================================
        # 2. ACTIVE CATEGORICAL VARIABLES
        # ============================================================
        X_cat_active = X[self.cat_cols_active_].astype(str)

        G_active = pd.get_dummies(
            X_cat_active,
            dtype=float,
        )

        # align with training modalities
        G_active = G_active.reindex(
            columns=self.active_modalities_,
            fill_value=0.0,
        )

        Z_cat_active = G_active.sub(
            self.active_modality_proportions_,
            axis=1,
        ).div(
            np.sqrt(self.active_modality_proportions_),
            axis=1,
        )

        # ============================================================
        # 3. BUILD ACTIVE Z MATRIX
        # ============================================================
        Z_active = pd.concat(
            [Z_num_active, Z_cat_active],
            axis=1,
        )

        # enforce exact training column order
        Z_active = Z_active.reindex(
            columns=self._Z_active_columns_,
        )

        # ============================================================
        # 4. PROJECT INTO PCA SPACE
        # ============================================================
        return super().row_coordinates(Z_active)

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
        """
        Cosine squared (quality of representation) of individuals
        on principal components.
        """

        F = self.row_coordinates(X)
        F = F**2
        denom = F.sum(axis=1)

        return F.div(denom, axis=0)

    @utils.check_is_fitted
    def numerical_column_correlations(self, include_supplementary=False):
        """Pearson correlations between quantitative variables (active and supplementary)
        and principal components.

        For quantitative variables, these values correspond to classical PCA loadings,
        i.e. correlations between standardized variables and axes.

        Correlations are not defined for categorical variables. For these variables,
        interpretation is based on contributions (η²) rather than correlations.

        For categorical modalities, coordinates correspond to barycenters of individuals
        carrying each modality (as in Multiple Correspondence Analysis, Pagès),
        and are therefore not interpreted as correlations.
        """
        if include_supplementary and hasattr(self, "num_cols_sup_") and self.num_cols_sup_:
            quanti = self.num_cols_active_ + self.num_cols_sup_
        else:
            quanti = self.num_cols_active_

        return self.column_coordinates_.loc[quanti]

    @utils.check_is_fitted
    def column_cosine_similarities(self):
        """
        Squared cosines (cos²) of variables and modalities with respect to principal components.

        This quantity measures the quality of representation of each element on each axis.

        Note that :
            - For quantitative variables, cos² is defined as the squared Pearson correlation
                between the standardized variable and each principal component.
            - For categorical variables, cos² is computed at the modality level using:
                cos² = (coordinate²) / eigenvalue

        These values are not directly defined at the variable level; categorical variables
        are therefore represented through their modalities.
        """
        # ------------------------------------------------------------
        # quantitative variables (PCA-style cos²)
        # ------------------------------------------------------------
        quanti_cos2 = self.numerical_column_correlations(True) ** 2

        # ------------------------------------------------------------
        # categorical variables (modality-based cos²)
        # ------------------------------------------------------------
        modality_cos2 = (self.modality_coordinates_active_**2).div(
            self.eigenvalues_,
            axis=1,
        )

        if hasattr(self, "modality_coordinates_sup_") and not self.modality_coordinates_sup_.empty:
            modality_cos2_sup = (self.modality_coordinates_sup_**2).div(
                self.eigenvalues_,
                axis=1,
            )
            modality_cos2 = pd.concat([modality_cos2, modality_cos2_sup], axis=0)

        # ------------------------------------------------------------
        # concatenate
        # ------------------------------------------------------------
        cos2 = pd.concat(
            [
                quanti_cos2,
                modality_cos2,
            ],
            axis=0,
        )

        ordered_index = self.column_coordinates_.index

        return cos2.loc[ordered_index]

    @utils.check_is_fitted
    def column_contributions(self, aggregate=True):
        """Column contributions.

        If aggregate=True:
            returns contributions at variable level (quanti + categorical η²)

        If aggregate=False:
            returns contributions at modality level for categorical variables
            + quantitative variables contributions.

        Note that :
            - For quantitative variables, the contribution to a principal components is defined as
                (coordinate²) / eigenvalue.
            - For modalities variables, the contribution to a principal components is defined using:
                proportion of this modality * (coordinate²) / eigenvalue.
            - For aggregated categorical variable, it is computed by η².

        """
        # ============================================================
        # 1. QUANTITATIVE VARIABLES
        # ============================================================
        quanti_ctr = (self.numerical_column_correlations() ** 2).div(self.eigenvalues_, axis=1)

        # ============================================================
        # 2. CATEGORICAL VARIABLES (MODALITIES)
        # ============================================================
        modality_coords = self.modality_coordinates_active_

        modality_freq = self.active_modality_proportions_

        modality_ctr = (
            (modality_coords**2).mul(modality_freq, axis=0).div(self.eigenvalues_, axis=1)
        )

        # ============================================================
        # 3. RETURN MODALITY LEVEL
        # ============================================================
        if not aggregate:
            return pd.concat(
                [
                    quanti_ctr,
                    modality_ctr,
                ]
            )

        # ============================================================
        # 4. AGGREGATION TO VARIABLE LEVEL (η²)
        # ============================================================
        cat_ctr = {}

        for var in self.cat_cols_active_:
            mods = [m for m in self.active_modalities_ if m.startswith(var + "_")]

            cat_ctr[var] = modality_ctr.loc[mods].sum()

        cat_ctr = pd.DataFrame.from_dict(cat_ctr, orient="index")

        # ============================================================
        # 5. CONCAT RESULT
        # ============================================================
        return pd.concat(
            [
                quanti_ctr,
                cat_ctr,
            ]
        )
