import unittest

import numpy as np
import pandas as pd

from prince import PCA


class TestPCA(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        # Load a dataframe
        dataframe = pd.read_csv('tests/data/decathlon.csv', index_col=0)

        # Determine the categorical columns
        cls.df_categorical = dataframe.select_dtypes(exclude=[np.number])

        # Determine the numerical columns
        cls.df_numeric = dataframe.drop(cls.df_categorical.columns, axis='columns')

        # Determine the covariance matrix
        X = cls.df_numeric.copy()
        cls.center_reduced = ((X - X.mean()) / X.std()).values
        cls.cov = cls.center_reduced.T @ cls.center_reduced

        # Calculate a full PCA
        cls.n_components = len(cls.df_numeric.columns)
        cls.pca = PCA(dataframe, n_components=cls.n_components, scaled=True)

    def test_dimensions_matches(self):
        self.assertEqual(self.pca.X.shape, self.df_numeric.shape)

    def test_column_names(self):
        self.assertListEqual(list(self.pca.X.columns), list(self.df_numeric.columns))

    def test_center_reduced(self):
        self.assertTrue(np.allclose(self.pca.X, self.center_reduced))

    def test_eigenvectors_dimensions_matches(self):
        self.assertEqual(self.pca.svd.U.shape, (self.df_numeric.shape[0], self.n_components))
        self.assertEqual(self.pca.svd.s.shape, (self.n_components,))
        self.assertEqual(self.pca.svd.V.shape, (self.n_components, self.df_numeric.shape[1]))

    def test_eigenvectors_form_orthonormal_basis(self):
        for i, v1 in enumerate(self.pca.svd.V.T):
            for j, v2 in enumerate(self.pca.svd.V.T):
                if i == j:
                    self.assertTrue(np.isclose(np.dot(v1, v2), 1))
                else:
                    self.assertTrue(np.isclose(np.dot(v1, v2), 0))

    def test_number_of_eigenvalues(self):
        self.assertEqual(len(self.pca.eigenvalues), self.n_components)

    def test_eigenvalues_sorted_desc(self):
        self.assertListEqual(self.pca.eigenvalues, list(reversed(sorted(self.pca.eigenvalues))))

    def test_eigenvalues_sum_equals_total_inertia(self):
        self.assertTrue(np.isclose(sum(self.pca.eigenvalues), self.pca.total_inertia))

    def test_eigenvalues_equals_squared_singular_values(self):
        for eigenvalue, singular_value in zip(self.pca.eigenvalues, self.pca.svd.s):
            self.assertTrue(np.isclose(eigenvalue, np.square(singular_value)))

    def test_cov_trace_equals_total_inertia(self):
        self.assertTrue(np.isclose(self.cov.trace(), self.pca.total_inertia))

    def test_eigenvalues_equals_column_inertias(self):
        col_inertias = np.square(self.pca.row_principal_coordinates).sum(axis='rows')
        for eigenvalue, col_inertia in zip(self.pca.eigenvalues, col_inertias):
            self.assertTrue(np.isclose(eigenvalue, col_inertia))

    def test_explained_inertia_sorted_desc(self):
        self.assertListEqual(
            self.pca.explained_inertia,
            list(reversed(sorted(self.pca.explained_inertia)))
        )

    def test_explained_inertia_sum(self):
        self.assertTrue(np.isclose(sum(self.pca.explained_inertia), 1))

    def test_cumulative_explained_inertia(self):
        self.assertListEqual(
            self.pca.cumulative_explained_inertia,
            list(np.cumsum(self.pca.explained_inertia))
        )

    def test_row_components_inertia(self):
        """Check the inertia of each row projection is equal to the associated eigenvalue and that
        the covariance between each row projection is nil."""
        for i, (_, p1) in enumerate(self.pca.row_principal_coordinates.iteritems()):
            for j, (_, p2) in enumerate(self.pca.row_principal_coordinates.iteritems()):
                if i == j:
                    self.assertTrue(np.isclose(
                        p1.var() * self.pca.total_inertia / self.n_components,
                        self.pca.eigenvalues[i])
                    )
                else:
                    self.assertTrue(np.isclose(np.cov(p1, p2)[0][1], 0))

    def test_row_components_contributions_sum_equals_total_inertia(self):
        for _, col_sum in self.pca.row_component_contributions.sum(axis='rows').iteritems():
            self.assertTrue(np.isclose(col_sum, 1))

    def test_row_cosine_similarities_shape_matches(self):
        self.assertEqual(
            self.pca.row_cosine_similarities.shape,
            (self.df_numeric.shape[0], self.n_components)
        )

    def test_row_cosine_similarities_are_bounded(self):
        n_cells = self.df_numeric.shape[0] * self.n_components
        self.assertEqual((-1 <= self.pca.row_cosine_similarities).sum().sum(), n_cells)
        self.assertEqual((self.pca.row_cosine_similarities <= 1).sum().sum(), n_cells)

    def test_column_correlations_shape_matches(self):
        self.assertEqual(
            self.pca.column_correlations.shape,
            (self.df_numeric.shape[1], self.n_components)
        )

    def test_column_correlations_bounded(self):
        n_cells = self.df_numeric.shape[1] * self.n_components
        self.assertEqual((-1 <= self.pca.column_correlations).sum().sum(), n_cells)
        self.assertEqual((self.pca.column_correlations <= 1).sum().sum(), n_cells)
