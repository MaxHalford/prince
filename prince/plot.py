from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy import stats

GRAY = OrderedDict([("light", "#bababa"), ("dark", "#404040")])


def stylize_axis(ax, grid=True):
    if grid:
        ax.grid()

    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    ax.axhline(y=0, linestyle="-", linewidth=1.2, color=GRAY["dark"], alpha=0.6)
    ax.axvline(x=0, linestyle="-", linewidth=1.2, color=GRAY["dark"], alpha=0.6)

    return ax


def confidence_ellipse(x, y, level=0.95, n_points=100):
    """Return the coordinates of the confidence ellipse around two arrays of numbers.

    The ellipse is centred on the mean of the points and shaped by their covariance matrix. Its
    radius is the `level` quantile of the chi-squared distribution with 2 degrees of freedom, so
    it covers `level` of the points, not of their mean, when they are bivariate normal.

    Args:
        x (1D array_like)
        y (1D array_like)
        level (float): The confidence level.
        n_points (int): The number of points used to draw the ellipse.

    Returns:
        tuple of 1D numpy arrays: The x and y coordinates of the ellipse.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(x, y))
    radius = np.sqrt(stats.chi2.ppf(level, df=2))

    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack((np.cos(t), np.sin(t)))
    ellipse = eigenvectors @ (np.sqrt(eigenvalues.clip(min=0))[:, None] * circle) * radius

    return x.mean() + ellipse[0], y.mean() + ellipse[1]
