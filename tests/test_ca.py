import numpy as np
import pandas as pd
import pytest

from prince import CA
from tests import util as test_util


@pytest.fixture
def df():
    """The original dataframe."""
    return pd.read_csv('tests/data/presidentielles07.csv', index_col=0)


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
def N(df):
    """The total number of observed value."""
    return np.sum(df.values)


@pytest.fixture
def ca(df, k):
    """The executed CA."""
    return CA(df, nbr_components=k)


def test_dimensions(ca, n, p):
    """Check the dimensions are correct."""
    assert ca.X.shape == (n, p)
    assert ca.P.shape == (n, p)


def test_eigenvectors_dimensions(ca, n, p, k):
    """Check the eigenvectors have the expected dimensions."""
    assert ca.svd.U.shape == (n, k)
    assert ca.svd.s.shape == (k,)
    assert ca.svd.V.shape == (k, p)


def test_total_sum(ca, N):
    """Check the total number of values is correct."""
    assert ca.N == N


def test_frequencies(ca, N, df):
    """Check the frequencies sums up to 1 and that the original data can be obtained by
    multiplying the frequencies by `N`."""
    assert np.isclose(ca.P.sum().sum(), 1)
    assert np.allclose(ca.P * N, df)


def test_row_sums_sum(ca):
    """Check the row sums sum up to 1."""
    assert np.isclose(ca.row_sums.sum(), 1)


def test_row_sums_shape(ca, n):
    """Check the row sums is a vector of length `n`."""
    assert ca.row_sums.shape == (n,)


def test_column_sums_sum(ca):
    """Check the column sums sum up to 1."""
    assert np.isclose(ca.column_sums.sum(), 1)


def test_column_sums_shape(ca, p):
    """Check the row sums is a vector of length `p`."""
    assert ca.column_sums.shape == (p,)


def test_expected_frequencies_shape(ca, n, p):
    """Check the expected frequencies matrix is of shape `(n, p)`."""
    assert ca.expected_frequencies.shape == (n, p)


def test_expected_frequencies_sum(ca, n, p):
    """Check the expected frequencies matrix sums to 1."""
    assert np.isclose(np.sum(ca.expected_frequencies.values), 1)


def test_eigenvalues_dimensions(ca, k):
    """Check the eigenvalues is a vector of length `k`."""
    assert len(ca.eigenvalues) == k


def test_eigenvalues_sorted(ca):
    """Check the eigenvalues are sorted in descending order."""
    assert test_util.is_sorted(ca.eigenvalues)


def test_eigenvalues_total_inertia(ca):
    """Check the eigenvalues sums to the same amount as the total inertia."""
    assert np.isclose(sum(ca.eigenvalues), ca.total_inertia, rtol=10e-3)


def test_eigenvalues_singular_values(ca):
    """Check the eigenvalues are the squares of the singular values."""
    for eigenvalue, singular_value in zip(ca.eigenvalues, ca.svd.s):
        assert np.isclose(eigenvalue, np.square(singular_value))


def test_explained_inertia_decreases(ca):
    """Check the explained inertia decreases."""
    assert test_util.is_sorted(ca.explained_inertia)


def test_explained_inertia_sum(ca):
    """Check the explained inertia sums to 1."""
    assert np.isclose(sum(ca.explained_inertia), 1, rtol=10e-3)


def test_cumulative_explained_inertia(ca):
    """Check the cumulative explained inertia is correct."""
    assert np.array_equal(ca.cumulative_explained_inertia, np.cumsum(ca.explained_inertia))


def test_row_component_contributions(ca, k):
    """Check the sum of row contributions is equal to the total inertia."""
    for _, col_sum in ca.row_component_contributions.sum(axis='rows').iteritems():
        assert np.isclose(col_sum, 1)


def test_row_cosine_similarities_shape(ca, n, k):
    """Check the shape of the variable correlations is coherent."""
    assert ca.row_cosine_similarities.shape == (n, k)


def test_row_cosine_similarities_bounded(ca, n, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= ca.row_cosine_similarities).sum().sum() == n * k
    assert (ca.row_cosine_similarities <= 1).sum().sum() == n * k


def test_row_profiles_shape(ca, n, p):
    """Check the row profiles is a matrix of shape `(n, p)`."""
    assert ca.row_profiles.shape == (n, p)


def test_row_profiles_sum(ca, n):
    """Check the row profiles sum up to 1 for each row."""
    for _, row_sum in ca.row_profiles.sum(axis='columns').iteritems():
        assert np.isclose(row_sum, 1)


def test_column_component_contributions(ca, k):
    """Check the sum of column contributions is equal to the total inertia."""
    for _, col_sum in ca.column_component_contributions.sum(axis='columns').iteritems():
        assert np.isclose(col_sum, 1)


def test_column_cosine_similarities_shape(ca, p, k):
    """Check the shape of the variable correlations is coherent."""
    assert ca.column_cosine_similarities.shape == (p, k)


def test_column_cosine_similarities_bounded(ca, p, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= ca.column_cosine_similarities).sum().sum() == p * k
    assert (ca.column_cosine_similarities <= 1).sum().sum() == p * k


def test_column_profiles_shape(ca, n, p):
    """Check the column profiles is a matrix of shape `(n, p)`."""
    assert ca.column_profiles.shape == (n, p)


def test_column_profiles_sum(ca, n):
    """Check the column profiles sum up to 1 for each column."""
    for _, column_sum in ca.column_profiles.sum(axis='rows').iteritems():
        assert np.isclose(column_sum, 1)
