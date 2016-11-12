from itertools import groupby
from operator import itemgetter

import numpy as np


def calculate_benzecri_correction(eigenvalues):
    """Because MCA and FAMD clouds often have a high dimensionality, the variance rates of the first
    principal components may be quite low, which makes them hard to interpret. Benzécri (1977) and
    Greenacre (1990) proposed to use modified rates to better appreciate the relative importance of
    the principal components."""
    p = len(eigenvalues)
    return [(p / (p - 1) * (eig - 1 / p)) ** 2 if eig > 1 / p else 0 for eig in eigenvalues]


def rescale(series, new_min=0, new_max=1):
    """Rescale the values of a `pandas.Series` from `new_min` to `new_max` using the min-max
    method."""
    old_min = series.min()
    old_max = series.max()
    series = series.apply(
        lambda x: (x - old_min) / (old_max - old_min) * \
                  (new_max - new_min) + new_min
    )
    return series


def cosine_similarity(A, B):
    """Calculate the cosine similarity between two arrays.

    Args:
        A (1D array_like)
        B (1D array_like)

    Returns:
        float: The cosine similarity between `A` and `B`.
    """
    return np.dot(A, B) / (np.sqrt(np.dot(A, A)) * np.sqrt(np.dot(B, B)))


def correlation_ratio(A, B):
    """Calculate the correlation ratio between a categorical array and a numerical array one.

    Args:
        A (1D array_like): contains categorial data.
        B (1D array_like): contains numerical data.

    Returns:
        float: The correlation ratio between `A` and `B`.
    """
    groups = groupby(sorted(zip(A, B), key=itemgetter(0)), key=itemgetter(0))
    means = []
    variances = []
    sizes = []
    for _, group in groups:
        values = [x for _, x in group]
        means.append(np.mean(values))
        variances.append(np.var(values))
        sizes.append(len(values))

    n = sum(sizes) # Total number of values, is also equal to the length of `A` and `B`
    mean = np.mean(means) # Global mean
    # The inter-group variance is the weighted mean of the group means
    var_inter = sum((s * (m - mean) ** 2 for s, m in zip(sizes, means))) / n
    # The intra-group variance is the weighted variance of the group variances
    var_intra = sum((s * v for s, v in zip(sizes, variances))) / n
    return var_inter / (var_inter + var_intra)
