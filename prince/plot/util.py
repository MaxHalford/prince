import numpy as np

from prince.svd import SVD


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
    svd = SVD(cov_matrix, k=2)

    chi_95 = np.sqrt(4.61) # 90% quantile of the chi-square distribution
    width = np.sqrt(cov_matrix[0][0]) * chi_95 * 2
    height = np.sqrt(cov_matrix[1][1]) * chi_95 * 2

    eigenvector = svd.V.T[0]
    angle = np.arctan(eigenvector[1] / eigenvector[0])

    return x_mean, y_mean, width, height, angle
