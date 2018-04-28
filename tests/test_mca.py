"""Tests for multiple correspondence analysis. The example comes from
https://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
"""
import unittest

import pandas as pd


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
