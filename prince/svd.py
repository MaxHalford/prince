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
            state = None
            if random_state is not None:
                state = np.random.get_state()
                np.random.seed(random_state)
            U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
            if state is not None:
                np.random.set_state(state)
        else:
            raise ValueError("fbpca is not installed; please install it if you want to use it")
    elif engine == "scipy":
        U, s, V = scipy.linalg.svd(X, full_matrices=False)
    elif engine == "sklearn":
        U, s, V = extmath.randomized_svd(
            X, n_components=n_components, n_iter=n_iter, random_state=random_state
        )
    else:
        raise ValueError("engine has to be one of ('fbpca', 'scipy', 'sklearn')")

    # Normalise to exactly ``n_components``. ``scipy.linalg.svd`` (and ``fbpca``) cap
    # at ``min(M, N)`` for rank reasons, while ``sklearn.randomized_svd`` always returns
    # the requested number, padding the tail with noise. We unify on the sklearn shape
    # contract — extra components past the matrix rank get zero singular values, and the
    # corresponding U/V columns are zeros (downstream code multiplies them by 0 anyway).
    k = len(s)
    if k < n_components:
        pad = n_components - k
        s = np.concatenate([s, np.zeros(pad)])
        U = np.hstack([U, np.zeros((U.shape[0], pad))])
        V = np.vstack([V, np.zeros((pad, V.shape[1]))])
    elif k > n_components:
        s = s[:n_components]
        U = U[:, :n_components]
        V = V[:n_components, :]

    # U, V = extmath.svd_flip(U, V)

    if row_weights is not None:
        U = U / np.sqrt(row_weights)[:, np.newaxis]  # row-wise scaling
    if column_weights is not None:
        V = V / np.sqrt(column_weights)

    return SVD(U, s, V)
