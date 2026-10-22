from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import rpy2.rinterface_lib
from rpy2.robjects import r as R
from sklearn.base import clone

# SciPy and FactoMineR use independent BLAS/LAPACK implementations. GitHub runner
# changes have produced absolute differences up to 4.18e-6 while the relative error
# remains small. Keep NumPy's strict relative tolerance and allow only that measured
# cross-backend numerical noise.
FACTOMINER_RTOL = 1e-7
FACTOMINER_ATOL = 5e-6

DATAFRAME_IMPLEMENTATIONS = (nw.Implementation.PANDAS, nw.Implementation.POLARS)


def _as_dataframe_implementation(X, implementation):
    if implementation is nw.Implementation.PANDAS:
        return X
    if implementation is nw.Implementation.POLARS:
        return pl.DataFrame({column: X[column].to_numpy() for column in X.columns})
    raise ValueError(f"Unsupported DataFrame implementation: {implementation}")


def assert_dataframe_implementation_matches(
    estimator, X_pandas, implementation, *, fit_params=None
):
    """Check that an estimator gives the same result for a DataFrame implementation."""
    fit_params = fit_params or {}
    X = _as_dataframe_implementation(X_pandas, implementation)
    expected_estimator = clone(estimator).fit(X_pandas, **fit_params)
    actual_estimator = clone(estimator).fit(X, **fit_params)
    method = "transform" if hasattr(estimator, "transform") else "row_coordinates"
    expected = getattr(expected_estimator, method)(X_pandas)
    actual = getattr(actual_estimator, method)(X)
    np.testing.assert_allclose(np.abs(actual.to_numpy()), np.abs(expected.to_numpy()))


def assert_allclose_to_factominer(actual, desired, *, rtol=FACTOMINER_RTOL):
    np.testing.assert_allclose(
        actual,
        desired,
        rtol=rtol,
        atol=FACTOMINER_ATOL,
    )


def load_df_from_R(code):
    df = R(code)
    if isinstance(df.names, rpy2.rinterface_lib.sexp.NULLType):
        return pd.DataFrame(np.array(df))
    return pd.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])
