from __future__ import annotations

import math
import tempfile

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as robjects
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import r as R
from scipy import sparse

import prince
from tests import load_df_from_R


@pytest.mark.parametrize(
    "sup_rows, sup_cols",
    [
        pytest.param(
            sup_rows,
            sup_cols,
            id=":".join(["sup_rows" if sup_rows else "", "sup_cols" if sup_cols else ""]).strip(
                ":"
            ),
        )
        for sup_rows in [False, True]
        for sup_cols in [False, True]
    ],
)
class TestCA:
    _row_name = "row"
    _col_name = "col"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5

        # Fit Prince
        self.dataset = prince.datasets.load_french_elections()
        active = self.dataset.copy()
        if sup_rows:
            active = active.drop("Île-de-France")
        if self.sup_cols:
            active = active.drop(columns=["Abstention", "Blank"])
        self.ca = prince.CA(n_components=n_components)
        self.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            self.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=1)")

        args = f"dataset, ncp={n_components}, graph=F"
        if self.sup_cols:
            if sup_rows:
                R(f"ca <- CA({args}, col.sup=c(13, 14), row.sup=c(18))")
            else:
                R(f"ca <- CA({args}, col.sup=c(13, 14))")
        else:
            if sup_rows:
                R(f"ca <- CA({args}, row.sup=c(18))")
            else:
                R(f"ca <- CA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.ca, prince.CA)
        sklearn.utils.validation.check_is_fitted(self.ca)

    def test_svd_U(self):
        F = load_df_from_R("ca$svd$U").to_numpy()
        P = sparse.diags(self.ca.row_masses_.to_numpy() ** -0.5) @ self.ca.svd_.U
        np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_svd_V(self):
        F = load_df_from_R("ca$svd$V").to_numpy()
        P = sparse.diags(self.ca.col_masses_.to_numpy() ** -0.5) @ self.ca.svd_.V.T
        np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_total_inertia(self):
        F = robjects.r("sum(ca$eig[,1])")[0]
        P = self.ca.total_inertia_
        assert math.isclose(F, P)

    def test_eigenvalues(self):
        F = load_df_from_R("ca$eig")[: self.ca.n_components]
        P = self.ca._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    def test_row_coords(self, method_name="row_coordinates"):
        F = load_df_from_R(f"ca${self._row_name}$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R(f"ca${self._row_name}.sup$coord")))

        method = getattr(self.ca, method_name)
        P = method(self.dataset)

        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R(f"ca${self._row_name}$contrib")
        P = self.ca.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_row_cosine_similarities(self):
        F = load_df_from_R(f"ca${self._row_name}$cos2")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R(f"ca${self._row_name}.sup$cos2")))
        P = self.ca.row_cosine_similarities(self.dataset)
        np.testing.assert_allclose(F, P)

    def test_col_coords(self):
        F = load_df_from_R(f"ca${self._col_name}$coord")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R(f"ca${self._col_name}.sup$coord")))
        P = self.ca.column_coordinates(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_contrib(self):
        F = load_df_from_R(f"ca${self._col_name}$contrib")
        P = self.ca.column_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_col_cos2(self):
        F = load_df_from_R(f"ca${self._col_name}$cos2")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R(f"ca${self._col_name}.sup$cos2")))
        P = self.ca.column_cosine_similarities(self.dataset)
        np.testing.assert_allclose(F, P)
