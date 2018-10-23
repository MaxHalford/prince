"""This module contains a custom one-hot encoder. It inherits from sklearn's
OneHotEncoder and returns a pandas.SparseDataFrame with appropriate column
names and index values.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn import preprocessing


class OneHotEncoder(preprocessing.OneHotEncoder):

    def __init__(self):
        super().__init__(sparse=True, dtype=np.uint8)

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X must be a pandas.DataFrame')

        self = super().fit(X)
        self.column_names_ = list(itertools.chain(*[
            [
                '{}_{}'.format(col, cat)
                for cat in self.categories_[i]
            ]
            for i, col in enumerate(X.columns)
        ]))

        return self

    def transform(self, X):
        return pd.SparseDataFrame(
            data=super().transform(X),
            columns=self.column_names_,
            index=X.index if isinstance(X, pd.DataFrame) else None,
            default_fill_value=0
        )
