from collections import OrderedDict

import numpy as np
from scipy import linalg


GRAY = OrderedDict([
    ('light', '#bababa'),
    ('dark', '#404040')
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
    U, s, V = linalg.svd(cov_matrix, full_matrices=False)

    chi_95 = np.sqrt(4.61)  # 90% quantile of the chi-square distribution
    width = np.sqrt(cov_matrix[0][0]) * chi_95 * 2
    height = np.sqrt(cov_matrix[1][1]) * chi_95 * 2

    eigenvector = V.T[0]
    angle = np.arctan(eigenvector[1] / eigenvector[0])

    return x_mean, y_mean, width, height, angle
