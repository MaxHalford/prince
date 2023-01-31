import tempfile
import numpy as np
import pandas as pd
import prince
import rpy2.rinterface_lib
from rpy2.robjects import r as R
from scipy import sparse
import sklearn.utils.estimator_checks
import sklearn.utils.validation

from tests import load_df_from_R


class FAMDTestSuite:
    _row_name = "row"
    _col_name = "col"
    sup_rows = False
    sup_cols = False

    @classmethod
    def setup_class(cls):

        n_components = 5

        # Fit Prince
        cls.dataset = prince.datasets.load_beers().head(1000)
        active = cls.dataset.copy()
        cls.famd = prince.FAMD(n_components=n_components, engine="scipy")
        cls.famd.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            cls.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=c(1))")
            R(f"famd <- FAMD(dataset, graph=F)")

    def test_check_is_fitted(self):
        assert isinstance(self.famd, prince.FAMD)
        sklearn.utils.validation.check_is_fitted(self.famd)

    def test_num_cols(self):
        assert self.famd.num_cols_ == [
            "alcohol_by_volume",
            "international_bitterness_units",
            "standard_reference_method",
            "final_gravity",
        ]

    def test_cat_cols(self):
        assert self.famd.cat_cols_ == ["style", "is_organic"]

    def test_eigenvalues(self):
        F = load_df_from_R("famd$eig")[: self.famd.n_components]
        P = self.famd._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    # def test_group_eigenvalues(self):

    #     for i, group in enumerate(self.groups, start=1):
    #         F = load_df_from_R(f"mfa$separate.analyses$Gr{i}$eig")[
    #             : self.mfa.n_components
    #         ]
    #         P = self.mfa[group]._eigenvalues_summary
    #         np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
    #         np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
    #         np.testing.assert_allclose(
    #             F["cumulative percentage of variance"], P["% of variance (cumulative)"]
    #         )

    # def test_row_coords(self):
    #     F = load_df_from_R(f"mfa$ind$coord")
    #     P = self.mfa.row_coordinates(self.dataset)
    #     np.testing.assert_allclose(F.abs(), P.abs())

    # def test_row_contrib(self):
    #     F = load_df_from_R("mfa$ind$contrib")
    #     P = self.mfa.row_contributions_
    #     np.testing.assert_allclose(F, P * 100)


class TestFAMDNoSup(FAMDTestSuite):
    ...
