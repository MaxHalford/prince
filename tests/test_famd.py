"""Tests for factor analysis of mixed data."""
import unittest

import numpy as np
import pandas as pd

import prince


class TestFAMD(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(
            data=[
                ['A', 'A', 'A', 2, 5, 7, 0, 3, 6, 7],
                ['A', 'A', 'A', 4, 4, 4, 0, 4, 4, 3],
                ['B', 'A', 'B', 5, 2, 1, 0, 7, 1, 1],
                ['B', 'A', 'B', 7, 2, 1, 0, 2, 2, 2],
                ['B', 'B', 'B', 3, 5, 6, 0, 2, 6, 6],
                ['B', 'B', 'A', 3, 5, 4, 0, 1, 7, 5]
            ],
            columns=['E1 fruity', 'E1 woody', 'E1 coffee',
                     'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
                     'E3 fruity', 'E3 butter', 'E3 woody'],
            index=['Wine {}'.format(i + 1) for i in range(6)]
        )

    def test_fit_pandas_dataframe(self):
        famd = prince.FAMD()
        self.assertTrue(isinstance(famd.fit(self.X), prince.FAMD))

    def test_only_numerical(self):
        famd = prince.FAMD()
        X = self.X.select_dtypes(np.number)
        with self.assertRaises(ValueError):
            famd.fit(X)
            famd.transform(X)

    def test_only_numerical_numpy(self):
        famd = prince.FAMD()
        X = self.X.select_dtypes(np.number)
        with self.assertRaises(ValueError):
            famd.fit(X.to_numpy())
            famd.transform(X.to_numpy())

    def test_only_categorical(self):
        famd = prince.FAMD()
        X = self.X.select_dtypes(exclude=np.number)
        with self.assertRaises(ValueError):
            famd.fit(X)
            famd.transform(X)
