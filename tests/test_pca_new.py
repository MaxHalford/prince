import numpy as np
import pandas as pd
import prince
import rpy2.robjects as robjects
import sklearn.utils.validation

R = robjects.r


def load_df_from_R(code):
    df = R(code)
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])


class PCATestSuite:
    use_sup_rows = False
    use_sup_cols = False

    @classmethod
    def setup_class(cls):
        cls.pca = prince.PCA()
        cls.pca.fit(prince.datasets.load_decathlon())

        R(
            """
        library('FactoMineR')

        data(decathlon)
        pca = PCA(decathlon[,1:10], ncp=5, graph=F)
        """
        )

    def test_pca_is_fitted(self):
        assert isinstance(self.pca, prince.PCA)
        sklearn.utils.validation.check_is_fitted(self.pca)

    def test_eigenvalues(self):
        eig = load_df_from_R("pca$eig")
        assert isinstance(eig, pd.DataFrame)


class TestWithNoSupData(PCATestSuite):
    ...


class TestWithSupRows(PCATestSuite):
    use_sup_rows = True

    def test_one(self):
        assert self.use_sup_rows
