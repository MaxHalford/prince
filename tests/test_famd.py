from __future__ import annotations

import tempfile

import numpy as np
import pytest
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import r as R

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
        for sup_rows in [False]
        for sup_cols in [False]
    ],
)
class TestFAMD:
    _row_name = "row"
    _col_name = "col"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5

        # Fit Prince
        self.dataset = prince.datasets.load_beers().head(200)
        active = self.dataset.copy()
        self.famd = prince.FAMD(n_components=n_components, engine="scipy")
        self.famd.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            self.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=c(1))")
            R("famd <- FAMD(dataset, graph=F)")

    def test_check_is_fitted(self):
        assert isinstance(self.famd, prince.FAMD)
        sklearn.utils.validation.check_is_fitted(self.famd)

    def test_num_cols(self):
        assert sorted(self.famd.num_cols_) == [
            "alcohol_by_volume",
            "final_gravity",
            "international_bitterness_units",
            "standard_reference_method",
        ]

    def test_cat_cols(self):
        assert sorted(self.famd.cat_cols_) == ["is_organic", "style"]

    def test_eigenvalues(self):
        F = load_df_from_R("famd$eig")[: self.famd.n_components]
        P = self.famd._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.famd, method_name)
        F = load_df_from_R("famd$ind$coord")
        P = method(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R("famd$ind$contrib")
        P = self.famd.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_col_coords(self):
        F = load_df_from_R("famd$var$coord")
        P = self.famd.column_coordinates_
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_contrib(self):
        F = load_df_from_R("famd$var$contrib")
        P = self.famd.column_contributions_
        np.testing.assert_allclose(F, P * 100)


def test_issue_169():
    """

    https://github.com/MaxHalford/prince/issues/169

    >>> import pandas as pd
    >>> from prince import FAMD
    >>> df = pd.DataFrame({'var1':['c', 'a', 'b','c'], 'var2':['x','y','y','z'],'var2': [0.,10.,30.4,0.]})

    >>> famd = FAMD(n_components=2, random_state=42)
    >>> famd = famd.fit(df[:3])

    >>> famd.transform(df[0:3])
    component         0         1
    0         -1.303760 -0.658334
    1         -0.335621  0.981047
    2          1.639381 -0.322713

    >>> famd.transform(df[0:2])
    component         0         1
    0         -1.000920 -0.669274
    1         -0.092001  0.669274

    >>> famd.transform(df[3:]).round(6)
    component         0    1
    3         -0.869173 -0.0

    """
