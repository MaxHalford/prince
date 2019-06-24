"""Tests for multiple correspondence analysis. The example comes from
https://www.utdallas.edu/~herve/Abdi-MFA2007-pretty.pdf
"""
import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd

import prince


class TestMFA(unittest.TestCase):

    def setUp(self):
        X = pd.DataFrame(
            data=[
                [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
                [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
                [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
                [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
                [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
                [3, 4, 4, 3, 5, 4, 5, 1, 7, 5]
            ],
            columns=['E1 fruity', 'E1 woody', 'E1 coffee',
                     'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
                     'E3 fruity', 'E3 butter', 'E3 woody'],
            index=['Wine {}'.format(i+1) for i in range(6)]
        )
        X = (X - X.mean()).apply(lambda x: x / np.sqrt((x ** 2).sum()), axis='rows')
        X['Oak type'] = [1, 2, 2, 2, 1, 1]
        self.X = X
        self.groups = {
            'Expert #{}'.format(no+1): [c for c in X.columns if c.startswith('E{}'.format(no+1))]
            for no in range(3)
        }

    def test_fit_pandas_dataframe(self):
        mfa = prince.MFA(groups=self.groups)
        self.assertTrue(isinstance(mfa.fit(self.X), prince.MFA))

    def test_transform_pandas_dataframe(self):
        mfa = prince.MFA(groups=self.groups)
        self.assertTrue(isinstance(mfa.fit(self.X).transform(self.X), pd.DataFrame))

    def test_plot_partial_row_coordinates(self):
        mfa = prince.MFA(groups=self.groups)
        for col in ['E1 fruity', 'E1 woody', 'E1 coffee']:
            self.X[col] = self.X[col].astype(str)
        mfa.fit(self.X)
        ax = mfa.plot_partial_row_coordinates(self.X)
        self.assertTrue(isinstance(ax, mpl.axes.Axes))

    def test_fit_numpy_array(self):
        groups = {
            name: [self.X.columns.get_loc(col) for col in cols]
            for name, cols in self.groups.items()
        }
        mfa = prince.MFA(groups=groups)
        self.assertTrue(isinstance(mfa.fit(self.X.values), prince.MFA))

    def test_transform_numpy_array(self):
        groups = {
            name: [self.X.columns.get_loc(col) for col in cols]
            for name, cols in self.groups.items()
        }
        mfa = prince.MFA(groups=groups)
        self.assertTrue(isinstance(mfa.fit(self.X.values).transform(self.X.values), pd.DataFrame))

    def test_no_groups(self):
        mfa = prince.MFA()
        with self.assertRaises(ValueError):
            mfa.fit(self.X)

    def test_mixed_groups(self):
        mfa = prince.MFA(groups=self.groups)
        self.X['E1 fruity'] = self.X['E1 fruity'].astype('category')
        with self.assertRaises(ValueError):
            mfa.fit(self.X)
