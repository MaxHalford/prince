"""Tests for multiple correspondence analysis. The example comes from
https://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
"""
import unittest

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

    def test_fit_pandas_dataframe(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X), prince.MCA))

    def test_transform_pandas_dataframe(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X).transform(self.X), pd.DataFrame))

    def test_fit_numpy_array(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X.values), prince.MCA))

    def test_transform_numpy_array(self):
        mca = prince.MCA(n_components=2)
        self.assertTrue(isinstance(mca.fit(self.X.values).transform(self.X.values), pd.DataFrame))
