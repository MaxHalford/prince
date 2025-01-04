from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as robjects
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import numpy2ri
from sklearn import decomposition, pipeline, preprocessing

import prince
from tests import load_df_from_R


@pytest.mark.parametrize(
    "sup_rows, sup_cols, scale, sample_weights, column_weights",
    [
        pytest.param(
            sup_rows,
            sup_cols,
            scale,
            sample_weights,
            column_weights,
            id=f"{sup_rows=}:{sup_cols=}:{scale=}:{sample_weights=}:{column_weights=}",
        )
        for sup_rows in [False, True]
        for sup_cols in [False, True]
        for scale in [False, True]
        for sample_weights in [False, True]
        for column_weights in [False, True]
    ],
)
class TestPCA:
    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols, scale, sample_weights, column_weights):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols
        self.scale = scale

        n_components = 5

        # Fit Prince
        self.dataset = prince.datasets.load_decathlon()
        self.active = self.dataset.copy()
        if self.sup_rows:
            self.active = self.active.query('competition == "Decastar"')
        self.sample_weights = (
            np.random.default_rng().dirichlet([1] * len(self.active)) if sample_weights else None
        )
        supplementary_columns = ["rank", "points"] if self.sup_cols else []
        self.column_weights = (
            np.random.default_rng().random(
                len(self.active.columns.difference(supplementary_columns))
            )
            if column_weights
            else None
        )
        self.pca = prince.PCA(n_components=n_components, rescale_with_std=self.scale)
        self.pca.fit(
            self.active,
            sample_weight=self.sample_weights,
            column_weight=self.column_weights,
            supplementary_columns=supplementary_columns,
        )

        # scikit-learn
        if self.scale:
            self.sk_pca = pipeline.make_pipeline(
                preprocessing.StandardScaler(),
                decomposition.PCA(n_components=n_components),
            )
        else:
            self.sk_pca = pipeline.make_pipeline(
                decomposition.PCA(n_components=n_components),
            )
        # sklearn's PCA doesn't support sample weights
        self.sk_pca.fit(self.active[self.pca.feature_names_in_])

        # Fit FactoMineR
        robjects.r(
            """
        library('FactoMineR')

        data(decathlon)
        decathlon <- subset(decathlon, select = -c(Competition))
        """
        )

        args = f"decathlon, ncp={n_components}, graph=F"
        if sample_weights:
            robjects.r.assign("row.w", numpy2ri.py2rpy(self.sample_weights))
            robjects.r("row.w <- as.vector(row.w)")
            args += ", row.w=row.w"
        if column_weights:
            robjects.r.assign("col.w", numpy2ri.py2rpy(self.column_weights))
            robjects.r("col.w <- as.vector(col.w)")
            args += ", col.w=col.w"
        if not self.scale:
            args += ", scale.unit=F"
        if self.sup_cols:
            if self.sup_rows:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12), ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12))")
        else:
            if self.sup_rows:
                robjects.r(f"pca = PCA({args}, ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.pca, prince.PCA)
        sklearn.utils.validation.check_is_fitted(self.pca)

    def test_total_inertia(self):
        F = robjects.r("sum(pca$eig[,1])")[0]
        P = self.pca.total_inertia_
        assert math.isclose(F, P)

    def test_eigenvalues(self):
        P = self.pca._eigenvalues_summary
        # Test against FactoMineR
        F = load_df_from_R("pca$eig")[: self.pca.n_components]
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )
        # Test against scikit-learn
        if self.sample_weights is None and self.column_weights is None:
            n = len(self.active)
            S = self.sk_pca[-1].explained_variance_ * (n - 1) / n
            np.testing.assert_allclose(P["eigenvalue"], S)
            np.testing.assert_allclose(
                P["% of variance"], self.sk_pca[-1].explained_variance_ratio_ * 100
            )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.pca, method_name)
        P = method(self.dataset)
        # Test againt FactoMineR
        F = load_df_from_R("pca$ind$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("pca$ind.sup$coord")))
        np.testing.assert_allclose(F.abs(), P.abs())
        # Test against scikit-learn
        if self.sample_weights is None and self.column_weights is None:
            S = self.sk_pca.transform(self.dataset[self.pca.feature_names_in_])
            np.testing.assert_allclose(np.abs(S), P.abs())

    def test_row_cosine_similarities(self):
        F = load_df_from_R("pca$ind$cos2")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("pca$ind.sup$cos2")))
        P = self.pca.row_cosine_similarities(self.dataset)
        np.testing.assert_allclose(F, P)

    def test_row_contrib(self):
        F = load_df_from_R("pca$ind$contrib")
        P = self.pca.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_col_coords(self):
        F = load_df_from_R("pca$var$coord")
        P = self.pca.column_coordinates_
        if self.sup_cols:
            P = P.drop(["rank", "points"])
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_cos2(self):
        F = load_df_from_R("pca$var$cos2")
        P = self.pca.column_cosine_similarities_
        if self.sup_cols:
            P = P.drop(["rank", "points"])
        np.testing.assert_allclose(F, P)

    def test_col_contrib(self):
        F = load_df_from_R("pca$var$contrib")
        P = self.pca.column_contributions_
        np.testing.assert_allclose(F, P * 100)
