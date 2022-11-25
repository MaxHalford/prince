import numpy as np
import pandas as pd
import prince
import rpy2.robjects as robjects
import sklearn.utils.validation


def load_df_from_R(code):
    df = robjects.r(code)
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])


class PCATestSuite:
    use_sup_rows = False
    use_sup_cols = False

    @classmethod
    def setup_class(cls):

        # Fit Prince
        dataset = prince.datasets.load_decathlon()
        cls.pca = prince.PCA(n_components=len(dataset.columns))
        cls.pca.fit(dataset)

        # Fit FactoMineR
        robjects.r(
            """
        library('FactoMineR')

        data(decathlon)
        pca = PCA(decathlon[,1:12], ncp=12, graph=F)
        """
        )

    def test_pca_is_fitted(self):
        assert isinstance(self.pca, prince.PCA)
        sklearn.utils.validation.check_is_fitted(self.pca)

    def test_eigenvalues(self):
        F = load_df_from_R("pca$eig")
        P = self.pca._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )


class TestPCANoSup(PCATestSuite):
    ...


class TestPCASupRows(PCATestSuite):
    use_sup_rows = True

    def test_one(self):
        assert self.use_sup_rows
