"""Tests for multiple correspondence analysis. The example comes from
https://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
"""
import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd

import prince


class TestMCA(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(
            data=[
                ['N', 'C', 'Y', 'N', 'Y', 'C', 'Y', 'Y', 'Y', 'Y'],
                ['Y', 'B', 'N', 'Y', 'N', 'B', 'N', 'Y', 'N', 'N'],
                ['Y', 'A', 'N', 'Y', 'N', 'A', 'N', 'Y', 'N', 'N'],
                ['Y', 'A', 'N', 'Y', 'N', 'A', 'N', 'N', 'N', 'N'],
                ['N', 'C', 'Y', 'N', 'Y', 'C', 'Y', 'N', 'Y', 'Y'],
                ['N', 'B', 'Y', 'N', 'Y', 'B', 'Y', 'N', 'Y', 'Y']
            ],
            columns=['E1 fruity', 'E1 woody', 'E1 coffee',
                     'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
                     'E3 fruity', 'E3 butter', 'E3 woody'],
            index=[1, 2, 3, 4, 5, 6]
        )

    def test_pandas_dataframe(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X), prince.MCA))
        self.assertTrue(isinstance(mca.transform(self.X), pd.DataFrame))

    def test_numpy_array(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X.to_numpy()), prince.MCA))
        self.assertTrue(isinstance(mca.transform(self.X.to_numpy()), pd.DataFrame))

    def test_plot_show_row_labels(self):
        mca = prince.MCA(n_components=2)
        mca.fit(self.X)
        ax = mca.plot_coordinates(self.X, show_row_labels=True)
        self.assertTrue(isinstance(ax, mpl.axes.Axes))

    def test_plot_show_column_labels(self):
        mca = prince.MCA(n_components=2)
        mca.fit(self.X)
        ax = mca.plot_coordinates(self.X, show_column_labels=True)
        self.assertTrue(isinstance(ax, mpl.axes.Axes))

    def test_fit_with_K(self):
        mca = prince.MCA(n_components=2, random_state=42)
        mca.fit(pd.get_dummies(self.X), K=10)
        self.assertEquals(mca.K, 10)

    def test_eigenvalues_are_corrected(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        self.assertEquals(mca.K, 10)
        np.testing.assert_allclose(mca.eigenvalues_, [.7004, .0123, .0003, 0 ], atol=0.0001)

    def test_total_inertia(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        np.testing.assert_almost_equal(mca.total_inertia_, 0.7130, 4)    

    def test_explained_inertia(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        self.assertEquals(mca.J, 22)
        np.testing.assert_allclose(mca.explained_inertia_, [.9519, .0168, .0004, 0 ], atol=0.0001)

    def test_row_contributions(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        r_cont = mca.row_contributions()
        pd.testing.assert_index_equal(r_cont.index, mca.row_masses_.index)
        np.testing.assert_allclose(r_cont.sum(axis=0), [1., 1., 1., 1. ], atol=0.0001)

    def test_column_contributions(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        c_cont = mca.column_contributions()
        pd.testing.assert_index_equal(c_cont.index, mca.col_masses_.index)
        np.testing.assert_allclose(c_cont.sum(axis=0), [1., 1., 1., 1. ], atol=0.0001)

    def test_row_cos2(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        r_cos2 = mca.row_cos2()
        self.assertEquals(r_cos2.shape, (6, 4))
        pd.testing.assert_index_equal(r_cos2.index, mca.row_masses_.index)
        self.assert_(np.all((r_cos2 >= 0) & (r_cos2 <= 1)), "All Cos2 should be between 0 and 1")
        self.assert_(np.all(r_cos2.sum(axis=1) <= 1), "Cos2 across dimensions should be near 1")

    def test_column_cos2(self):
        mca = prince.MCA(n_components=4, random_state=42)
        mca.fit(self.X)
        c_cos2 = mca.column_cos2()
        self.assertEquals(c_cos2.shape, (22, 4))
        pd.testing.assert_index_equal(c_cos2.index, mca.col_masses_.index)
        self.assert_(np.all((c_cos2 >= 0) & (c_cos2 <= 1)), "All Cos2 should be between 0 and 1")
        # Should be really <= 1., but account for floating precision error
        self.assert_(np.all(c_cos2.sum(axis=1) <= 1.000001), "Cos2 across dimensions should be near 1")
