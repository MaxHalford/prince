"""Tests for correspondence analysis. The hair/eye data comes from section
17.2.3 of http://ce.aut.ac.ir/~shiry/lecture/Advanced%20Machine%20Learning/Manifold_Modern_Multivariate%20Statistical%20Techniques%20-%20Regres.pdf
"""
import unittest

import pandas as pd

import prince


class TestCA(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(
            data=[
                [326, 38, 241, 110, 3],
                [688, 116, 584, 188, 4],
                [343, 84, 909, 412, 26],
                [98, 48, 403, 681, 85]
            ],
            columns=pd.Series(['Fair', 'Red', 'Medium', 'Dark', 'Black']).rename('Hair color'),
            index=pd.Series(['Blue', 'Light', 'Medium', 'Dark']).rename('Eye color')
        )

    def test_fit_numpy_array(self):
        ca = prince.CA(n_components=2)
        self.assertTrue(isinstance(ca.fit(self.X.values), prince.CA))

    def test_transform_numpy_array(self):
        ca = prince.CA(n_components=2)
        self.assertTrue(isinstance(ca.fit(self.X.values).transform(self.X.values), pd.DataFrame))

    def test_fit_pandas_dataframe(self):
        ca = prince.CA(n_components=2)
        self.assertTrue(isinstance(ca.fit(self.X), prince.CA))

    def test_transform_pandas_dataframe(self):
        ca = prince.CA(n_components=2)
        self.assertTrue(isinstance(ca.fit(self.X).transform(self.X), pd.DataFrame))

    def test_negative_input(self):
        ca = prince.CA()
        self.X.iloc[0, 0] = -1
        with self.assertRaises(ValueError):
            ca.fit(self.X)
