"""Singular Value Decomposition (SVD)"""
import dataclasses

try:
    import fbpca

    FBPCA_INSTALLED = True
except ImportError:
    FBPCA_INSTALLED = False
import numpy as np
from sklearn.utils import extmath


@dataclasses.dataclass
class SVD:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray


def compute_svd(X, n_components, n_iter, random_state, engine) -> SVD:
    """Computes an SVD with k components."""

    # TODO: support sample weights

    # Compute the SVD
    if engine == "fbpca":
        if FBPCA_INSTALLED:
            U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
        else:
            raise ValueError(
                "fbpca is not installed; please install it if you want to use it"
            )
    elif engine == "sklearn":
        U, s, V = extmath.randomized_svd(
            X, n_components=n_components, n_iter=n_iter, random_state=random_state
        )
    else:
        raise ValueError("engine has to be one of ('fbpca', 'sklearn')")

    U, V = extmath.svd_flip(U, V)

    return SVD(U, s, V)
