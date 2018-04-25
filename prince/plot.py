from collections import OrderedDict

import numpy as np

from . import svd


GRAY = OrderedDict([
    ('light', '#bababa'),
    ('dark', '#404040')
])

SEABORN = OrderedDict([
    ('blue', '#4c72b0'),
    ('green', '#55a868'),
    ('red', '#c44e52'),
    ('yellow', '#ccb974'),
    ('purple', '#8172b2'),
    ('cyan', '#64b5cd')
])


def stylize_axis(ax, grid=True):

    if grid:
        ax.grid()

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)
    ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)

    return ax


def build_ellipse(X, Y):
    """Construct ellipse coordinates from two arrays of numbers.

    Args:
        X (1D array_like)
        Y (1D array_like)

    Returns:
        float: The mean of `X`.
        float: The mean of `Y`.
        float: The width of the ellipse.
        float: The height of the ellipse.
        float: The angle of orientation of the ellipse.

    """
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    cov_matrix = np.cov(X, Y)
    U, s, V = svd.compute_svd(cov_matrix, n_components=2, n_iter=20, engine='auto')

    chi_95 = np.sqrt(4.61)  # 90% quantile of the chi-square distribution
    width = np.sqrt(cov_matrix[0][0]) * chi_95 * 2
    height = np.sqrt(cov_matrix[1][1]) * chi_95 * 2

    eigenvector = V.T[0]
    angle = np.arctan(eigenvector[1] / eigenvector[0])

    return x_mean, y_mean, width, height, angle
