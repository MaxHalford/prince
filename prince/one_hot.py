"""
This module contains a custom one-hot encoder. It inherits from sklearn's OneHotEncoder and returns
a pandas.DataFrame with sparse values.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn import preprocessing


class OneHotEncoder(preprocessing.OneHotEncoder):

    def __init__(self):
        super().__init__(sparse=True, dtype=np.uint8, categories='auto')

    def fit(self, X, y=None):

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

        oh = pd.DataFrame.sparse.from_spmatrix(super().transform(X))
        oh.columns = self.column_names_
        if isinstance(X, pd.DataFrame):
            oh.index = X.index

        return oh
