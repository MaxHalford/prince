import numpy as np
import pandas as pd
import pytest

from prince import PCA
from tests import util as test_util


@pytest.fixture
def load_df():
    """The original dataframe."""
    return pd.read_csv('tests/data/decathlon.csv', index_col=0)


@pytest.fixture
def df(load_df):
    """The original dataframe without categorical columns."""
    valid_columns = [
        column
        for column in load_df.columns
        if load_df[column].dtype in ('int64', 'float64')
    ]
    return load_df[valid_columns]


@pytest.fixture
def n(df):
    """The number of rows."""
    n, _ = df.shape
    return n


@pytest.fixture
def p(df):
    """The number of columns."""
    _, p = df.shape
    return p


@pytest.fixture
def k(p):
    """The number of principal components to compute."""
    return p


@pytest.fixture
def pca(load_df, k):
    """The executed PCA."""
    return PCA(load_df, nbr_components=k, scaled=True)


@pytest.fixture
def cov_matrix(df):
    """The covariance matrix of the filtered original dataframe."""
    df -= df.mean()
    df /= df.std()
    X = df.values
    return X.T @ X


def test_categorical(pca, df):
    """Check the categorical variable has been ignored."""
    assert np.array_equal(pca.X.columns, df.columns)


def test_dimensions(pca, n, p):
    """Check the dimensions are correct."""
    assert pca.X.shape == (n, p)


def test_center_reduce(pca, df):
    """Check the data was centered and reduced."""
    assert np.allclose(pca.X, (df - df.mean()) / df.std())


def test_eigenvectors_dimensions(pca, n, p, k):
    """Check the eigenvectors have the expected dimensions."""
    assert pca.svd.U.shape == (n, k)
    assert pca.svd.s.shape == (k,)
    assert pca.svd.V.shape == (k, p)


def test_eigenvectors_orthnormal(pca):
    """Check the eigenvectors define an orthonormal base."""
    for i, v1 in enumerate(pca.svd.V.T):
        for j, v2 in enumerate(pca.svd.V.T):
            if i == j:
                assert np.isclose(np.dot(v1, v2), 1)
            else:
                assert np.isclose(np.dot(v1, v2), 0)


def test_eigenvalues_dimensions(pca, k):
    """Check the eigenvalues is a vector of length `k`."""
    assert len(pca.eigenvalues) == k


def test_eigenvalues_sorted(pca):
    """Check the eigenvalues are sorted in descending order."""
    assert test_util.is_sorted(pca.eigenvalues)


def test_eigenvalues_total_inertia(pca):
    """Check the eigenvalues sums to the same amount as the total inertia."""
    assert np.isclose(sum(pca.eigenvalues), pca.total_inertia)


def test_eigenvalues_singular_values(pca):
    """Check the eigenvalues are the squares of the singular values."""
    for eigenvalue, singular_value in zip(pca.eigenvalues, pca.svd.s):
        assert np.isclose(eigenvalue, np.square(singular_value))


def test_cov_trace_total_inertia(pca, cov_matrix):
    """Check the trace of the covariance matrix is equal to the total inertia."""
    assert np.isclose(cov_matrix.trace(), pca.total_inertia)


def test_eigenvalues_column_inertias(pca):
    """Check the eigenvalues are the sums of the column inertias."""
    squared_row_pc = np.square(pca.row_principal_components)
    col_inertias = squared_row_pc.sum(axis='rows')
    for eig, col_inertia in zip(pca.eigenvalues, col_inertias):
        assert np.isclose(eig, col_inertia)


def test_explained_inertia_decreases(pca):
    """Check the explained inertia decreases."""
    assert test_util.is_sorted(pca.explained_inertia)


def test_explained_inertia_sum(pca):
    """Check the explained inertia sums to 1."""
    assert np.isclose(sum(pca.explained_inertia), 1)


def test_cumulative_explained_inertia(pca):
    """Check the cumulative explained inertia is correct."""
    assert np.array_equal(pca.cumulative_explained_inertia, np.cumsum(pca.explained_inertia))


def test_row_components_variance(pca, k):
    """Check the variance of each row projection is equal to the associated eigenvalue and that the
    covariance between each row projection is nil."""
    eigenvalues = pca.eigenvalues

    for i, (_, p1) in enumerate(pca.row_principal_components.iteritems()):
        for j, (_, p2) in enumerate(pca.row_principal_components.iteritems()):
            if i == j:
                assert np.isclose(p1.var() * pca.total_inertia / k, eigenvalues[i])
            else:
                assert np.isclose(np.cov(p1, p2)[0][1], 0)


def test_row_components_contributions(pca, k):
    """Check the sum of row contributions is equal to the total inertia."""
    for _, col_sum in pca.row_component_contributions.sum(axis='rows').iteritems():
        assert np.isclose(col_sum, 1)


def test_row_cosine_similarities_shape(pca, n, k):
    """Check the shape of the variable cosines is coherent."""
    assert pca.row_cosine_similarities.shape == (n, k)


def test_row_cosine_similarities_bounded(pca, n, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= pca.row_cosine_similarities).sum().sum() == n * k
    assert (pca.row_cosine_similarities <= 1).sum().sum() == n * k


def test_column_correlations_shape(pca, p, k):
    """Check the shape of the variable correlations is coherent."""
    assert pca.column_correlations.shape == (p, k)


def test_column_correlations_bounded(pca, p, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= pca.column_correlations).sum().sum() == p * k
    assert (pca.column_correlations <= 1).sum().sum() == p * k
