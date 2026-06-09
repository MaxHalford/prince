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
        categorical_columns = categorical_columns or []
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
        self.active_categories_ = {
            col: X_cat_active[col].astype(str).unique() for col in self.active_categorical_columns_
        }

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

        # 10. Row coordinates
        self.row_coordinates_ = pd.DataFrame(
            self.svd_.U * np.sqrt(self.eigenvalues_),
            index=X.index,
        )
        self.row_coordinates_.columns.name = "component"

        # 11. Column coordinates
        self.column_coordinates_.columns.name = "component"

        # 12. Active numerical coordinates
        self.active_numerical_coordinates_ = self.column_coordinates_.loc[
            self.active_numerical_columns_
        ]
        self.active_numerical_coordinates_.index.name = "variable"

        # 13. Active categorical modality coordinates
        self.active_modality_coordinates_ = self.column_coordinates_.loc[self.active_modalities_]
        self.active_modality_coordinates_.index.name = "modality"

        # 14. Supplementary numerical coordinates
        self.supplementary_numerical_coordinates_ = pd.DataFrame()

        if self.supplementary_numerical_columns_:
            self.supplementary_numerical_coordinates_ = self.column_coordinates_.loc[
                self.supplementary_numerical_columns_
            ]
            self.supplementary_numerical_coordinates_.index.name = "variable"

        # 15. Supplementary categorical modality coordinates
        self.supplementary_modality_coordinates_ = pd.DataFrame()

        if self.supplementary_categorical_columns_:
            self.supplementary_modality_coordinates_ = self.column_coordinates_.loc[
                self.supplementary_categorical_modalities_
            ]
            self.supplementary_modality_coordinates_.index.name = "modality"

        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        """Return the principal coordinates of the rows.

        Only active variables are used to reconstruct the FAMD space.
        Supplementary variables do not participate in the projection:
        they are projected separately after the factor space is built.

        Arguments:
            X: Dataset 
        
        Returns:
            Dataframe of row coordinates
        """
        # Preprocessing of dataset using fitting scalers

        # Active numerical variables
        X_num_active = X[self.active_numerical_columns_].copy()

        Z_num_active = pd.DataFrame(
            self.active_numerical_scaler_.transform(X_num_active),
            index=X.index,
            columns=self.active_numerical_columns_,
        )

        # Active categorical variables
        X_cat_active = X[self.active_categorical_columns_].astype(str)

        G_active = pd.get_dummies(
            X_cat_active,
            dtype=float,
        )

        # Align with training modalities
        G_active = G_active.reindex(
            columns=self.active_modalities_,
            fill_value=0.0,
        )

        Z_cat_active = G_active.sub(
            self.active_modalities_proportions_,
            axis=1,
        ).div(
            np.sqrt(self.active_modalities_proportions_),
            axis=1,
        )

        # Build active Z matrix
        Z_active = pd.concat(
            [Z_num_active, Z_cat_active],
            axis=1,
        )

        # Project into PCA space
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
        """Cosine squared of individuals on principal components.
        
        Arguments:
            X: Dataset 
        
        Returns:
            Dataframe of row cosine similarities.
        """
        F = self.row_coordinates(X)
        F = F**2
        denom = F.sum(axis=1)

        return F.div(denom, axis=0)

    @utils.check_is_fitted
    def column_correlations(self):
        """Pearson correlations between quantitative variables and principal components.

        Correlation are given for active and supplementary variables.
        For quantitative variables, these values correspond to classical PCA loadings,
        i.e. correlations between standardized variables and axes.

        Since categorical modalities are encoded as numerical variables, one can compute their
        correlations with the principal components.
        These correlations correspond to the modality coordinates multiplied by
        the modality's standard deviation.

        Correlations are not defined for categorical variables. For these variables,
        interpretation is based on contributions (η²) rather than correlations.
        """
        # Actives
        # Numerical columns
        active_correlation_num = self.active_numerical_coordinates_

        # Modalities
        active_p = self.active_modalities_proportions_
        active_correlation_mod = self.active_modality_coordinates_.multiply(np.sqrt(1 - active_p), axis=0)
        correlations = pd.concat([active_correlation_num, active_correlation_mod])

        # Supplementary
        # Numerical columns
        if hasattr(self, "supplementary_numerical_columns_") and self.supplementary_numerical_columns_:
            supplementary_correlation_num = self.supplementary_numerical_coordinates_
            correlations = pd.concat([correlations, supplementary_correlation_num])

        # Modalities
        if hasattr(self, "supplementary_categorical_columns_") and self.supplementary_categorical_columns_:
            sup_p = self.supplementary_modalities_proportions_
            sup_correlation_mod = self.supplementary_modality_coordinates_.divided(np.sqrt(1 - sup_p), axis=0)
            correlations = pd.concat([correlations, sup_correlation_mod])

        return correlations

    @utils.check_is_fitted
    def column_cosine_similarities(self):
        """Squared cosines (cos²) of variables and modalities with respect to principal components.

        This quantity measures the quality of representation of each element on each axis.
        For quantitative variables and modalities, cos² is defined as the squared Pearson
        correlation between the standardized variable and each principal component.

        Notes:
            Cosine similarities are not directly defined for categorical variables.
        """
        return self.column_correlations() ** 2

    @utils.check_is_fitted
    def column_contributions(self, aggregate = True):
        """Column contributions.

        For quantitative variables:
            contribution = (coordinate²) / eigenvalue

        For modalities:
            contribution = (coordinate²) / eigenvalue

        If aggregate=True, categorical contributions are aggregated :
            η² / eigenvalue (Pagès définition of η²)

        Arguments:
            aggregate: If True, returns variable-level contributions.
                If False, returns modality-level contributions.

        Returns:
            DataFrame with contributions per component.
        """
        # Numerical columns
        quanti_ctr = (self.active_numerical_coordinates_**2).div(self.eigenvalues_, axis=1)

        # Modalities
        modality_ctr = (self.active_modality_coordinates_**2).div(self.eigenvalues_, axis=1)

        # Return modality level
        if not aggregate:
            return pd.concat(
                [
                    quanti_ctr,
                    modality_ctr,
                ]
            )

        # Aggregation to variable level
        cat_ctr = {}

        for var in self.active_categorical_columns_:
            mods = [m for m in self.active_modalities_ if m.startswith(var + "_")]
            cat_ctr[var] = modality_ctr.loc[mods].sum()
        cat_ctr = pd.DataFrame.from_dict(cat_ctr, orient="index")

        # Concat result
        return pd.concat(
            [
                quanti_ctr,
                cat_ctr,
            ]
        )