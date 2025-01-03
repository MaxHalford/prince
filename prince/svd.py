"""Singular Value Decomposition (SVD)"""

from __future__ import annotations

import dataclasses

try:
    import fbpca

    FBPCA_INSTALLED = True
except ImportError:
    FBPCA_INSTALLED = False
import numpy as np
import scipy
from sklearn.utils import extmath


@dataclasses.dataclass
class SVD:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray


def compute_svd(
    X: np.ndarray,
    n_components: int,
    n_iter: int,
    engine: str,
    random_state: int | None = None,
    row_weights: np.ndarray | None = None,
    column_weights: np.ndarray | None = None,
) -> SVD:
    """Computes an SVD with k components."""

    if row_weights is not None:
        X = X * np.sqrt(row_weights[:, np.newaxis])  # row-wise scaling
    if column_weights is not None:
        X = X * np.sqrt(column_weights)

    # Compute the SVD
    if engine == "fbpca":
        if FBPCA_INSTALLED:
            U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
        else:
            raise ValueError("fbpca is not installed; please install it if you want to use it")
    elif engine == "scipy":
        U, s, V = scipy.linalg.svd(X)
        U = U[:, :n_components]
        s = s[:n_components]
        V = V[:n_components, :]
    elif engine == "sklearn":
        U, s, V = extmath.randomized_svd(
            X, n_components=n_components, n_iter=n_iter, random_state=random_state
        )
    else:
        raise ValueError("engine has to be one of ('fbpca', 'scipy', 'sklearn')")

    # U, V = extmath.svd_flip(U, V)

    if row_weights is not None:
        U = U / np.sqrt(row_weights)[:, np.newaxis]  # row-wise scaling
    if column_weights is not None:
        V = V / np.sqrt(column_weights)

    return SVD(U, s, V)
