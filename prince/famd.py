"""Factor Analysis of Mixed Data (FAMD)"""
import numpy as np
import pandas as pd

from . import mfa


class FAMD(mfa.MFA):

    def __init__(self, n_components=2, n_iter=3, copy=True, random_state=None, engine='auto'):
        super().__init__(
            groups=None,
            normalize=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns.tolist()
        cat_cols = set(X.columns) - set(num_cols)

        # Make one per variable type
        self.groups = {
            'Numerical': num_cols,
            'Categorical': cat_cols
        }

        return super().fit(X)
