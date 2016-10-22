import numpy as np
import pandas as pd
import pytest

from prince import FAMD, util
from tests import util as test_util


@pytest.fixture
def df():
    """The original dataframe."""
    return pd.read_csv('tests/data/ogm.csv', index_col=0)


@pytest.fixture
def indicator_matrix(df):
    """The indicator matrix of the original dataframe."""
    indicator_matrix = pd.get_dummies(df)

    for column in df.columns:
        if df[column].dtype in ('int64', 'float64'):
            indicator_matrix[column] = util.rescale(df[column], new_min=0, new_max=1)

    return indicator_matrix


@pytest.fixture
def n(indicator_matrix):
    """The number of rows."""
    n, _ = indicator_matrix.shape
    return n


@pytest.fixture
def p(indicator_matrix):
    """The number of columns."""
    _, p = indicator_matrix.shape
    return p


@pytest.fixture
def k(p):
    """The number of principal components to compute."""
    return p


@pytest.fixture
def N(indicator_matrix):
    """The total number of observed value."""
    return np.sum(indicator_matrix.values)


@pytest.fixture
def famd(df, k):
    """The executed CA."""
    return FAMD(df, nbr_components=k)


def test_dimensions(famd, n, p):
    """Check the dimensions are correct."""
    assert famd.X.shape == (n, p)


def test_eigenvectors_dimensions(famd, n, p, k):
    """Check the eigenvectors have the expected dimensions."""
    assert famd.svd.U.shape == (n, k)
    assert famd.svd.s.shape == (k,)
    assert famd.svd.V.shape == (k, p)


def test_total_sum(famd, N):
    """Check the total number of values is correct."""
    assert famd.N == N


def test_frequencies(famd, N, indicator_matrix):
    """Check the frequencies sums up to 1 and that the original data famdn be obtained by
    multiplying the frequencies by N."""
    assert np.isclose(famd.P.sum().sum(), 1)
    assert np.allclose(famd.P * N, indicator_matrix)


def test_row_sums_sum(famd):
    """Check the row sums sum up to 1."""
    assert np.isclose(famd.row_sums.sum(), 1)


def test_row_sums_shape(famd, n):
    """Check the row sums is a vector of length n."""
    assert famd.row_sums.shape == (n,)


def test_column_sums_sum(famd):
    """Check the column sums sum up to 1."""
    assert np.isclose(famd.column_sums.sum(), 1)


def test_column_sums_shape(famd, p):
    """Check the row sums is a vector of length p."""
    assert famd.column_sums.shape == (p,)


def test_expected_frequencies_shape(famd, n, p):
    """Check the expected frequencies matrix is of shape (n, p)."""
    assert famd.expected_frequencies.shape == (n, p)


def test_expected_frequencies_sum(famd, n, p):
    """Check the expected frequencies matrix sums to 1."""
    assert np.isclose(np.sum(famd.expected_frequencies.values), 1)


def test_eigenvalues_dimensions(famd, k):
    """Check the eigenvalues is a vector of length k."""
    assert len(famd.eigenvalues) == k


def test_eigenvalues_sorted(famd):
    """Check the eigenvalues are sorted in descending order."""
    assert test_util.is_sorted(famd.eigenvalues)


def test_eigenvalues_total_inertia(famd):
    """Check the eigenvalues sums to the same amount as the total inertia."""
    assert np.isclose(sum(famd.eigenvalues), famd.total_inertia)


def test_eigenvalues_singular_values(famd):
    """Check the eigenvalues are the squares of the singular values."""
    for eigenvalue, singular_value in zip(famd.eigenvalues, famd.svd.s):
        assert np.isclose(eigenvalue, np.square(singular_value))


def test_explained_inertia_decreases(famd):
    """Check the explained inertia decreases."""
    assert test_util.is_sorted(famd.explained_inertia)


def test_explained_inertia_sum(famd):
    """Check the explained inertia sums to 1."""
    assert np.isclose(sum(famd.explained_inertia), 1)


def test_cumulative_explained_inertia(famd):
    """Check the cumulative explained inertia is correct."""
    assert np.array_equal(famd.cumulative_explained_inertia, np.cumsum(famd.explained_inertia))


def test_row_component_contributions(famd, k):
    """Check the sum of row contributions is equal to the total inertia."""
    for _, col_sum in famd.row_component_contributions.sum(axis='rows').iteritems():
        assert np.isclose(col_sum, 1)


def test_row_cosine_similarities_shape(famd, n, k):
    """Check the shape of the variable correlations is coherent."""
    assert famd.row_cosine_similarities.shape == (n, k)


def test_row_cosine_similarities_bounded(famd, n, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= famd.row_cosine_similarities).sum().sum() == n * k
    assert (famd.row_cosine_similarities <= 1).sum().sum() == n * k


def test_row_profiles_shape(famd, n, p):
    """Check the row profiles is a matrix of shape (n, p)."""
    assert famd.row_profiles.shape == (n, p)


def test_row_profiles_sum(famd, n):
    """Check the row profiles sum up to 1 for each row."""
    for _, row_sum in famd.row_profiles.sum(axis='columns').iteritems():
        assert np.isclose(row_sum, 1)


def test_column_component_contributions(famd, k):
    """Check the sum of column contributions is equal to the total inertia."""
    for _, col_sum in famd.column_component_contributions.sum(axis='columns').iteritems():
        assert np.isclose(col_sum, 1)


def test_column_cosine_similarities_shape(famd, p, k):
    """Check the shape of the variable correlations is coherent."""
    assert famd.column_cosine_similarities.shape == (p, k)


def test_column_cosine_similarities_bounded(famd, p, k):
    """Check the variable correlations are bounded between -1 and 1."""
    assert (-1 <= famd.column_cosine_similarities).sum().sum() == p * k
    assert (famd.column_cosine_similarities <= 1).sum().sum() == p * k


def test_column_profiles_shape(famd, n, p):
    """Check the column profiles is a matrix of shape `(n, p)`."""
    assert famd.column_profiles.shape == (n, p)


def test_column_profiles_sum(famd, n):
    """Check the column profiles sum up to 1 for each column."""
    for _, column_sum in famd.column_profiles.sum(axis='rows').iteritems():
        assert np.isclose(column_sum, 1)
