import numpy as np
import pandas as pd
import prince
import rpy2.rinterface_lib
import rpy2.robjects as robjects
import sklearn.utils.validation


def load_df_from_R(code):
    df = robjects.r(code)
    if isinstance(df.names, rpy2.rinterface_lib.sexp.NULLType):
        return pd.DataFrame(np.array(df))
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])


class PCATestSuite:
    sup_rows = False
    sup_cols = False

    @classmethod
    def setup_class(cls):

        # Fit Prince
        cls.dataset = prince.datasets.load_decathlon()
        active = cls.dataset.copy()
        if cls.sup_rows:
            active = active.query('competition == "Decastar"')
        n_components = 5
        cls.pca = prince.PCA(n_components=n_components)
        cls.pca.fit(
            active, supplementary_columns=["rank", "points"] if cls.sup_cols else None
        )

        # Fit FactoMineR
        robjects.r(
            f"""
        library('FactoMineR')

        data(decathlon)
        decathlon <- subset(decathlon, select = -c(Competition))
        """
        )
        args = f"decathlon, ncp={n_components}, graph=F"
        if cls.sup_cols:
            if cls.sup_rows:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12), ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12))")
        else:
            if cls.sup_rows:
                robjects.r(f"pca = PCA({args}, ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.pca, prince.PCA)
        sklearn.utils.validation.check_is_fitted(self.pca)

    def test_eigenvalues(self):
        F = load_df_from_R("pca$eig")[: self.pca.n_components]
        P = self.pca._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    def test_row_coords(self):
        F = load_df_from_R("pca$ind$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("pca$ind.sup$coord")))
        P = self.pca.row_coordinates(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_cos2(self):
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


class TestPCANoSup(PCATestSuite):
    ...


class TestPCASupRows(PCATestSuite):
    sup_rows = True


class TestPCASupCols(PCATestSuite):
    sup_cols = True


class TestPCASupRowsSupCols(PCATestSuite):
    sup_cols = True
