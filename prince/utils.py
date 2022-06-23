import functools
import pandas as pd
from sklearn.utils import validation


def check_is_fitted(method):
    """Decorator"""

    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        validation.check_is_fitted(self)
        return method(self, *method_args, **method_kwargs)

    return _impl


def make_labels_and_names(X):

    if isinstance(X, pd.DataFrame):
        row_label = X.index.name if X.index.name else "Rows"
        row_names = X.index.tolist()
        col_label = X.columns.name if X.columns.name else "Columns"
        col_names = X.columns.tolist()
    else:
        row_label = "Rows"
        row_names = list(range(X.shape[0]))
        col_label = "Columns"
        col_names = list(range(X.shape[1]))

    return row_label, row_names, col_label, col_names
