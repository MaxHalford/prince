"""Factor Analysis of Mixed Data (FAMD)"""
import collections
import itertools

import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn import preprocessing

from prince import pca
from prince import plot
from prince import utils


class FAMD(pca.PCA):
    def __init__(
        self,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        as_array=False,
    ):
        super().__init__(
            rescale_with_mean=True,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
            as_array=as_array,
        )

    def fit(self, X, y=None):

        # Separate numerical columns from categorical columns
        self.num_cols_ = X.select_dtypes(np.number).columns.tolist()
        if not self.num_cols_:
            raise ValueError("All variables are qualitative: MCA should be used")
        self.cat_cols_ = list(set(X.columns) - set(self.num_cols_))
        if not self.cat_cols_:
            raise ValueError("All variables are quantitative: PCA should be used")

        # Preprocess numerical columns
        X_num = X[self.num_cols_].copy()
        num_scaler_ = preprocessing.StandardScaler().fit(X_num)
        X_num[:] = num_scaler_.transform(X_num)

        # Preprocess categorical columns
        X_cat = X[self.cat_cols_]
        cat_scaler_ = preprocessing.OneHotEncoder().fit(X_cat)
        X_cat = pd.DataFrame.sparse.from_spmatrix(
            cat_scaler_.transform(X_cat),
            index=X_cat.index,
            columns=cat_scaler_.get_feature_names_out(self.cat_cols_),
        )
        prop = X_cat.sum() / X_cat.sum().sum() * 2
        X_cat = X_cat.sub(X_cat.mean(axis="rows")).div(prop**0.5, axis="columns")

        Z = pd.concat([X_num, X_cat], axis=1)

        return super().fit(Z)
