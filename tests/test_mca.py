import numpy as np
import pandas as pd
import pytest

from prince import MCA
from tests import util as test_util


@pytest.fixture
def df():
    """The original dataframe."""
    return pd.read_csv('tests/data/ogm.csv', index_col=0)


@pytest.fixture
def indicator_matrix(df):
    """The indicator matrix of the original dataframe."""
    return pd.get_dummies(df)


@pytest.fixture
def n(indicator_matrix):
    """The number of rows."""
    n, _ = indicator_matrix.shape
    return n


@pytest.fixture
def p(indicator_matrix):
    """The number of columns in the indicator matrix."""
    _, p = indicator_matrix.shape
    return p


@pytest.fixture
def q(df):
    """The number of columns in the initial dataframe."""
    _, q = df.shape
    return q


@pytest.fixture
def k(p):
    """The number of principal components to compute."""
    return p


@pytest.fixture
def N(indicator_matrix):
    """The total number of observed value."""
    return np.sum(indicator_matrix.values)


@pytest.fixture
def mca(df, k):
    """The executed CA."""
    return MCA(df, nbr_components=k)


def test_dimensions(mca, n, p):
    """Check the dimensions are correct."""
    assert mca.X.shape == (n, p)


def test_eigenvectors_dimensions(mca, n, p, k):
    """Check the eigenvectors have the expected dimensions."""
    assert mca.svd.U.shape == (n, k)
    assert mca.svd.s.shape == (k,)
    assert mca.svd.V.shape == (k, p)


def test_total_sum(mca, N):
    """Check the total number of values is correct."""
    assert mca.N == N


def test_frequencies(mca, N, indicator_matrix):
    """Check the frequencies sums up to 1 and that the original data mcan be obtained by
    multiplying the frequencies by N."""
    assert np.isclose(mca.P.sum().sum(), 1)
    assert np.allclose(mca.P * N, indicator_matrix)


def test_row_sums_sum(mca):
    """Check the row sums sum up to 1."""
    assert np.isclose(mca.row_sums.sum(), 1)


def test_row_sums_shape(mca, n):
    """Check the row sums is a vector of length `n`."""
    assert mca.row_sums.shape == (n,)


def test_column_sums_sum(mca):
    """Check the column sums sum up to 1."""
    assert np.isclose(mca.column_sums.sum(), 1)


def test_column_sums_shape(mca, p):
    """Check the row sums is a vector of length `p`."""
    assert mca.column_sums.shape == (p,)


def test_expected_frequencies_shape(mca, n, p):
    """Check the expected frequencies matrix is of shape `(n, p)`."""
    assert mca.expected_frequencies.shape == (n, p)


def test_expected_frequencies_sum(mca, n, p):
    """Check the expected frequencies matrix sums to 1."""
    assert np.isclose(np.sum(mca.expected_frequencies.values), 1)


def test_eigenvalues_dimensions(mca, k):
    """Check the eigenvalues is a vector of length `k`."""
    assert len(mca.eigenvalues) == k


def test_eigenvalues_sorted(mca):
    """Check the eigenvalues are sorted in descending order."""
    assert test_util.is_sorted(mca.eigenvalues)


def test_eigenvalues_total_inertia(mca):
    """Check the eigenvalues sums to the same amount as the total inertia."""
    assert np.isclose(sum(mca.eigenvalues), mca.total_inertia)


def test_eigenvalues_singular_values(mca):
    """Check the eigenvalues are the squares of the singular values."""
    for eigenvalue, singular_value in zip(mca.eigenvalues, mca.svd.s):
        assert np.isclose(eigenvalue, np.square(singular_value))


def test_explained_inertia_decreases(mca):
    """Check the explained inertia decreases."""
    assert test_util.is_sorted(mca.explained_inertia)


def test_explained_inertia_sum(mca):
    """Check the explained inertia sums to 1."""
    assert np.isclose(sum(mca.explained_inertia), 1)


def test_cumulative_explained_inertia(mca):
    """Check the cumulative explained inertia is correct."""
    assert np.array_equal(mca.cumulative_explained_inertia, np.cumsum(mca.explained_inertia))


def test_row_component_contributions(mca, k):
    """Check the sum of row contributions is equal to the total inertia."""
    for _, col_sum in mca.row_component_contributions.sum(axis='rows').iteritems():
        assert np.isclose(col_sum, 1)


def test_row_cosine_similarities_shape(mca, n, k):
    """Check the shape of the variable correlations is coherent."""
    assert mca.row_cosine_similarities.shape == (n, k)


def test_row_cosine_similarities_bounded(mca, n, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= mca.row_cosine_similarities).sum().sum() == n * k
    assert (mca.row_cosine_similarities <= 1).sum().sum() == n * k


def test_row_profiles_shape(mca, n, p):
    """Check the row profiles is a matrix of shape (n, p)."""
    assert mca.row_profiles.shape == (n, p)


def test_row_profiles_sum(mca, n):
    """Check the row profiles sum up to 1 for each row."""
    for _, row_sum in mca.row_profiles.sum(axis='columns').iteritems():
        assert np.isclose(row_sum, 1)


def test_column_component_contributions(mca, k):
    """Check the sum of column contributions is equal to the total inertia."""
    for _, col_sum in mca.column_component_contributions.sum(axis='columns').iteritems():
        assert np.isclose(col_sum, 1)


def test_column_cosine_similarities_shape(mca, p, k):
    """Check the shape of the variable correlations is coherent."""
    assert mca.column_cosine_similarities.shape == (p, k)


def test_column_cosine_similarities_bounded(mca, p, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= mca.column_cosine_similarities).sum().sum() == p * k
    assert (mca.column_cosine_similarities <= 1).sum().sum() == p * k


def test_column_profiles_shape(mca, n, p):
    """Check the column profiles is a matrix of shape `(n, p)`."""
    assert mca.column_profiles.shape == (n, p)


def test_column_profiles_sum(mca, n):
    """Check the column profiles sum up to 1 for each column."""
    for _, column_sum in mca.column_profiles.sum(axis='rows').iteritems():
        assert np.isclose(column_sum, 1)


def test_column_correlations_shape(mca, q, k):
    """Check the shape of the variable correlations is coherent."""
    assert mca.column_correlations.shape == (q, k)


def test_column_correlations_bounded(mca, q, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= mca.column_correlations).sum().sum() == q * k
    assert (mca.column_correlations <= 1).sum().sum() == q * k
