import pandas as pd
import numpy as np

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

def chi2_dist(u,v,DI):
    """
    Compute the chi squared distance as in formula (17.18) from Izenman's book
    u,v: row or column vectors of row or column profiles matrix 
    DI - inverse matrix of row or column masses
    """
    d = (u-v).T @ DI @ (u-v)
    return np.sqrt(d)
