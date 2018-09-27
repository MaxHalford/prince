import numba as nb
import pandas as pd


def make_labels_and_names(X):

    if isinstance(X, pd.DataFrame):
        row_label = X.index.name if X.index.name else 'Rows'
        row_names = X.index.tolist()
        col_label = X.columns.name if X.columns.name else 'Columns'
        col_names = X.columns.tolist()
    else:
        row_label = 'Rows'
        row_names = list(range(X.shape[0]))
        col_label = 'Columns'
        col_names = list(range(X.shape[1]))

    return row_label, row_names, col_label, col_names


@nb.jit(parallel=True, nogil=True)
def scale_transform(X, copy, with_mean, with_std):
    scaler_ = preprocessing.StandardScaler(
                copy=copy,
                with_mean=with_mean,
                with_std=with_std
            ).fit(X)
    X = scaler_.transform(X)
    return X