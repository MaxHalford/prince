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


@pytest.mark.parametrize("engine", ["fbpca", "sklearn"])
class TestRandomState:
    """random_state must produce reproducible SVDs regardless of the global
    numpy random state, for every engine that exposes a random_state knob."""

    X = np.random.RandomState(0).rand(500, 100)

    @pytest.fixture(autouse=True)
    def _skip_if_engine_unavailable(self, engine):
        if engine == "fbpca" and not svd.FBPCA_INSTALLED:
            pytest.skip("fbpca is not installed")

    def _svd(self, engine, random_state):
        return svd.compute_svd(
            X=self.X, n_components=5, n_iter=3, random_state=random_state, engine=engine
        )

    def test_same_seed_gives_identical_output(self, engine):
        r1 = self._svd(engine, random_state=42)
        np.random.rand(5)  # disturb the global numpy random state
        r2 = self._svd(engine, random_state=42)
        np.testing.assert_array_equal(r1.U, r2.U)
        np.testing.assert_array_equal(r1.s, r2.s)
        np.testing.assert_array_equal(r1.V, r2.V)

    def test_different_seeds_give_different_output(self, engine):
        r1 = self._svd(engine, random_state=42)
        r2 = self._svd(engine, random_state=1)
        assert not np.allclose(r1.U, r2.U)

    def test_global_random_state_is_not_mutated(self, engine):
        np.random.seed(99)
        expected = np.random.rand(3)

        np.random.seed(99)
        self._svd(engine, random_state=42)
        actual = np.random.rand(3)

        np.testing.assert_array_equal(expected, actual)
