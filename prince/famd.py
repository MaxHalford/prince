"""Factor Analysis of Mixed Data (FAMD)"""
import collections
import itertools

import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder

from prince import pca
from prince import plot
from prince import utils


class FAMD(pca.PCA):
    def fit(self, X, y=None):

        # Separate numerical columns from categorical columns
        self.num_cols_ = X.select_dtypes(np.number).columns.tolist()
        if not self.num_cols_:
            raise ValueError("All variables are qualitative: MCA should be used")
        self.cat_cols_ = list(set(X.columns) - set(self.num_cols_))
        if not self.cat_cols_:
            raise ValueError("All variables are quantitative: PCA should be used")

        # return super().fit(X)
