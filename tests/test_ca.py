import unittest

import numpy as np
import pandas as pd

from prince import CA


class TestCA(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        # Load a dataframe
        cls.initial_dataframe = pd.read_csv('tests/data/presidentielles07.csv', index_col=0)

        # Detemine the shape of the initial dataframe
        (cls.n, cls.p) = cls.initial_dataframe.shape

        # Determine the total number of observations
        cls.N = np.sum(cls.initial_dataframe.values)

        # Calculate a full CA
        cls.n_components = cls.p
        cls.ca = CA(cls.initial_dataframe, n_components=cls.n_components)

    def test_eigenvectors_dimensions_match(self):
        self.assertEqual(self.ca.svd.U.shape, (self.n, self.n_components))
        self.assertEqual(self.ca.svd.s.shape, (self.n_components,))
        self.assertEqual(self.ca.svd.V.shape, (self.n_components, self.p))

    def test_number_of_observations(self):
        self.assertEqual(self.ca.N, self.N)

    def test_frequencies(self):
        """Check the frequencies sums up to 1 and that the original data can be obtained by
        multiplying the frequencies by `N`."""
        self.assertTrue(np.isclose(self.ca.P.sum().sum(), 1))
        self.assertTrue(np.allclose(self.ca.P * self.ca.N, self.initial_dataframe))

    def test_row_sums_shape(self):
        self.assertEqual(self.ca.row_sums.shape, (self.n,))

    def test_row_sums_sum(self):
        self.assertTrue(np.isclose(self.ca.row_sums.sum(), 1))

    def test_column_sums_shape(self):
        self.assertEqual(self.ca.column_sums.shape, (self.p,))

    def test_column_sums_sum(self):
        self.assertTrue(np.isclose(self.ca.column_sums.sum(), 1))

    def test_expected_frequencies_shape(self):
        self.assertEqual(self.ca.expected_frequencies.shape, (self.n, self.p))

    def test_expected_frequencies_sum(self):
        self.assertTrue(np.isclose(np.sum(self.ca.expected_frequencies.values), 1))

    def test_number_of_eigenvalues(self):
        self.assertEqual(len(self.ca.eigenvalues), self.n_components)

    def test_eigenvalues_sorted_desc(self):
        self.assertListEqual(self.ca.eigenvalues, list(reversed(sorted(self.ca.eigenvalues))))

    def test_eigenvalues_sum_equals_total_inertia(self):
        self.assertTrue(np.isclose(sum(self.ca.eigenvalues), self.ca.total_inertia, rtol=10e-3))

    def test_eigenvalues_equals_squared_singular_values(self):
        for eigenvalue, singular_value in zip(self.ca.eigenvalues, self.ca.svd.s):
            self.assertTrue(np.isclose(eigenvalue, np.square(singular_value)))

    def test_explained_inertia_sorted_desc(self):
        self.assertListEqual(
            self.ca.explained_inertia,
            list(reversed(sorted(self.ca.explained_inertia)))
        )

    def test_explained_inertia_sum(self):
        self.assertTrue(np.isclose(sum(self.ca.explained_inertia), 1, rtol=10e-3))

    def test_cumulative_explained_inertia(self):
        self.assertListEqual(
            self.ca.cumulative_explained_inertia,
            list(np.cumsum(self.ca.explained_inertia))
        )

    def test_row_components_contributions_sum_equals_total_inertia(self):
            for _, col_sum in self.ca.row_component_contributions.sum(axis='rows').iteritems():
                self.assertTrue(np.isclose(col_sum, 1))

    def test_row_cosine_similarities_shape_matches(self):
        self.assertEqual(self.ca.row_cosine_similarities.shape, (self.n, self.n_components))

    def test_row_cosine_similarities_are_bounded(self):
        n_cells = self.n * self.n_components
        self.assertEqual((-1 <= self.ca.row_cosine_similarities).sum().sum(), n_cells)
        self.assertEqual((self.ca.row_cosine_similarities <= 1).sum().sum(), n_cells)

    def test_row_profiles_shape(self):
        self.assertEqual(self.ca.row_profiles.shape, (self.n, self.p))

    def test_row_profiles_sum(self):
        for _, row_sum in self.ca.row_profiles.sum(axis='columns').iteritems():
            self.assertTrue(np.isclose(row_sum, 1))

    def test_column_component_contributions(self):
        for _, col_sum in self.ca.column_component_contributions.sum(axis='columns').iteritems():
            self.assertTrue(np.isclose(col_sum, 1))

    def test_column_cosine_similarities_shape(self):
        self.assertEqual(self.ca.column_cosine_similarities.shape, (self.p, self.n_components))

    def test_column_cosine_similarities_bounded(self):
        n_cells = self.p * self.n_components
        self.assertEqual((-1 <= self.ca.column_cosine_similarities).sum().sum(), n_cells)
        self.assertEqual((self.ca.column_cosine_similarities <= 1).sum().sum(), n_cells)

    def test_column_profiles_shape(self):
        self.assertEqual(self.ca.column_profiles.shape, (self.n, self.p))

    def test_column_profiles_sum(self):
        for _, col_sum in self.ca.column_profiles.sum(axis='rows').iteritems():
            self.assertTrue(np.isclose(col_sum, 1))
