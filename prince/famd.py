"""Factor Analysis of Mixed Data (FAMD)"""
import numpy as np
import pandas as pd

from . import mfa


class FAMD(mfa.MFA):

    def __init__(self, n_components=2, n_iter=3, copy=True, check_input=True, random_state=None,
                 engine='auto'):
        super().__init__(
            groups=None,
            normalize=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns.tolist()
        cat_cols = list(set(X.columns) - set(num_cols))

        # Make one group per variable type
        self.groups = {}
        if num_cols:
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have categorical data; you should consider using MCA')
        if cat_cols:
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have numerical data; you should consider using PCA')

        return super().fit(X)
