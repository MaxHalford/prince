"""
Work in progress.
"""


import pandas as pd

from . import util
from .mca import MCA


class FAMD(MCA):

    """Factor Analysis of Mixed Data"""

    def _build_indicator_matrix(self):
        """Build the indicator matrix by placing a "1" where a row takes a value for a column and
        a "0" when it doesn't."""

        indicator_matrix = pd.get_dummies(self.X)

        # Add the numerical columns after rescaling
        for col in self.numerical_columns.columns:
            indicator_matrix[col] = util.rescale(
                series=self.numerical_columns[col],
                new_min=0,
                new_max=1
            )

        return indicator_matrix
