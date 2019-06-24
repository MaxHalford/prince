"""Tests for multiple correspondence analysis. The example comes from
https://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
"""
import unittest

import matplotlib as mpl
import pandas as pd

import prince


class TestMCA(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(
            data=[
                [0, 'C', 1, 0, 1, 'C', 1, 1, 1, 1],
                [1, 'B', 0, 1, 0, 'B', 0, 1, 0, 0],
                [1, 'A', 0, 1, 0, 'A', 0, 1, 0, 0],
                [1, 'A', 0, 1, 0, 'A', 0, 0, 0, 0],
                [0, 'C', 1, 0, 1, 'C', 1, 0, 1, 1],
                [0, 'B', 1, 0, 1, 'B', 1, 0, 1, 1]
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
