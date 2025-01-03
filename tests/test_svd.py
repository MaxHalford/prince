from __future__ import annotations

import numpy as np
import pytest
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from prince import svd
from tests import load_df_from_R


@pytest.mark.parametrize(
    "n_components, are_rows_weighted, are_columns_weighted",
    [
        pytest.param(
            n_components,
            are_rows_weighted,
            are_columns_weighted,
            id=f"{n_components=}:{are_rows_weighted=}:{are_columns_weighted=}",
        )
        for n_components in [1, 3, 10]
        for are_rows_weighted in [False, True]
        for are_columns_weighted in [False, True]
    ],
)
class TestSVD:
    @pytest.fixture(autouse=True)
    def _prepare(self, n_components, are_rows_weighted, are_columns_weighted):
        self.n_components = n_components
        self.are_rows_weighted = are_rows_weighted
        self.are_columns_weighted = are_columns_weighted

        self.dataset = np.random.rand(100, 10)
        self.row_weights = np.random.rand(100)
        self.row_weights /= self.row_weights.sum()
        self.column_weights = np.random.rand(10)

        # Fit Prince
        self.svd = svd.compute_svd(
            X=self.dataset,
            row_weights=self.row_weights if are_rows_weighted else None,
            column_weights=self.column_weights if are_columns_weighted else None,
            n_components=n_components,
            n_iter=3,
            random_state=42,
            engine="scipy",
        )

        # Fit FactoMineR
        robjects.r("library('FactoMineR')")
        robjects.r.assign("X", numpy2ri.py2rpy(self.dataset))
        robjects.r.assign("row.w", numpy2ri.py2rpy(self.row_weights))
        robjects.r.assign("col.w", numpy2ri.py2rpy(self.column_weights))
        robjects.r("row.w <- as.vector(row.w)")
        robjects.r("col.w <- as.vector(col.w)")
        args = f"X, ncp={n_components}"
        if are_rows_weighted:
            args += ", row.w=row.w"
        if are_columns_weighted:
            args += ", col.w=col.w"
        robjects.r(f"svd = svd.triplet({args})")

    def test_U(self):
        assert self.svd.U.shape == (100, self.n_components)
        if self.are_rows_weighted:
            P = self.svd.U
            F = load_df_from_R("svd$U")
            np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_s(self):
        assert self.svd.s.shape == (self.n_components,)
        if self.are_rows_weighted:
            P = self.svd.s
            F = robjects.r("svd$vs")[: self.n_components]
            np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_V(self):
        assert self.svd.V.shape == (self.n_components, 10)
        P = self.svd.V
        F = load_df_from_R("svd$V").T
        np.testing.assert_allclose(np.abs(F), np.abs(P))
