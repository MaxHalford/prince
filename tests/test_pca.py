import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import decomposition
from sklearn.utils import estimator_checks

import prince


class TestPCA(unittest.TestCase):

    def setUp(self):
        X, _ = datasets.load_iris(return_X_y=True)
        columns = ['Sepal length', 'Sepal width', 'Petal length', 'Sepal length']
        self.X = pd.DataFrame(X, columns=columns)

    def test_fit_pandas_dataframe(self):
        pca = prince.PCA(n_components=2, engine='fbpca')
        self.assertTrue(isinstance(pca.fit(self.X), prince.PCA))

    def test_transform_pandas_dataframe(self):
        pca = prince.PCA(n_components=2)
        self.assertTrue(isinstance(pca.fit(self.X).transform(self.X), pd.DataFrame))

    def test_fit_numpy_array(self):
        pca = prince.PCA(n_components=2, engine='fbpca')
        self.assertTrue(isinstance(pca.fit(self.X.values), prince.PCA))

    def test_fit_bad_engine(self):
        pca = prince.PCA(n_components=2, engine='caca')
        with self.assertRaises(ValueError):
            pca.fit(self.X)

    def test_transform_numpy_array(self):
        pca = prince.PCA(n_components=2)
        self.assertTrue(isinstance(pca.fit(self.X.values).transform(self.X.values), pd.DataFrame))

    def test_copy(self):
        XX = np.copy(self.X)

        pca = prince.PCA(n_components=2, copy=True)
        pca.fit(XX)
        np.testing.assert_array_equal(self.X, XX)

        pca = prince.PCA(n_components=2, copy=False)
        pca.fit(XX)
        self.assertRaises(AssertionError, np.testing.assert_array_equal, self.X, XX)

    def test_fit_transform(self):

        # Without rescaling
        prince_pca = prince.PCA(n_components=3, rescale_with_mean=False, rescale_with_std=False)
        pd.testing.assert_frame_equal(
            prince_pca.fit_transform(self.X),
            prince_pca.fit(self.X).transform(self.X)
        )

        # With rescaling
        prince_pca = prince.PCA(n_components=3, rescale_with_mean=True, rescale_with_std=True)
        pd.testing.assert_frame_equal(
            prince_pca.fit_transform(self.X),
            prince_pca.fit(self.X).transform(self.X)
        )

    def test_compare_sklearn(self):

        n_components = 4
        pca_prince = prince.PCA(n_components=n_components, rescale_with_std=False)
        pca_sklearn = decomposition.PCA(n_components=n_components)

        pca_prince.fit(self.X)
        pca_sklearn.fit(self.X)

        # Compare eigenvalues
        np.testing.assert_array_almost_equal(
            pca_prince.eigenvalues_,
            pca_sklearn.explained_variance_ * (len(self.X) - 1) / len(self.X),
        )

        # Compare row projections
        np.testing.assert_array_almost_equal(
            pca_prince.transform(self.X),
            pca_sklearn.transform(self.X)
        )

        # Compare explained inertia
        np.testing.assert_array_almost_equal(
            pca_prince.explained_inertia_,
            pca_sklearn.explained_variance_ratio_
        )

        # Compare inverse transforms
        np.testing.assert_array_almost_equal(
            pca_prince.inverse_transform(pca_prince.transform(self.X)),
            pca_sklearn.inverse_transform(pca_sklearn.transform(self.X)),
        )

    def test_explained_inertia_(self):
        pca = prince.PCA(n_components=4)
        pca.fit(self.X)
        self.assertTrue(np.isclose(sum(pca.explained_inertia_), 1))

    def test_plot_row_coordinates(self):
        pca = prince.PCA(n_components=4)
        pca.fit(self.X)
        ax = pca.plot_row_coordinates(self.X)
        self.assertTrue(isinstance(ax, mpl.axes.Axes))

    def test_plot_row_coordinates_numpy(self):
        pca = prince.PCA(n_components=4)
        pca.fit(self.X.to_numpy())
        ax = pca.plot_row_coordinates(self.X.to_numpy())
        self.assertTrue(isinstance(ax, mpl.axes.Axes))

    def test_check_estimator(self):
        estimator_checks.check_estimator(prince.PCA(as_array=True))

    def test_column_correlations_numpy(self):
        pca = prince.PCA()
        pca.fit(self.X.to_numpy())
        self.assertTrue(isinstance(pca.column_correlations(self.X.to_numpy()), pd.DataFrame))

    def test_row_standard_coordinates(self):
        pca = prince.PCA()
        pca.fit(self.X)
        self.assertTrue(isinstance(pca.row_standard_coordinates(self.X), pd.DataFrame))

    def test_row_cosine_similarities(self):
        pca = prince.PCA()
        pca.fit(self.X)
        self.assertTrue(isinstance(pca.row_cosine_similarities(self.X), pd.DataFrame))
