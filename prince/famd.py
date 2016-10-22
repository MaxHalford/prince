import pandas as pd

from . import util
from .mca import MCA


class FAMD(MCA):

    """Factor Analysis of Mixed Data"""

    def _build_indicator_matrix(self):
        """Build the indicator matrix by placing a "1" where a row takes a value for a variable and
        a "0" when it doesn't."""

        indicator_matrix = pd.get_dummies(self.X)

        # Add the numerical variables after rescaling
        for col in self.numerical_variables.columns:
            indicator_matrix[col] = util.rescale(
                series=self.numerical_variables[col],
                new_min=0,
                new_max=1
            )

        return indicator_matrix
