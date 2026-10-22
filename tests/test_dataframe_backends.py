from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import prince


def _cases():
    rng = np.random.default_rng(42)

    ca = pd.DataFrame(
        [[12, 5, 8], [4, 15, 6], [9, 7, 11]],
        columns=["a", "b", "c"],
    )
    mca = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue", "green"],
            "size": ["s", "m", "l", "s", "l", "m"],
        }
    )
    famd = pd.DataFrame(
        {
            "height": [1.60, 1.72, 1.68, 1.81, 1.75, 1.65],
            "weight": [52.0, 70.0, 63.0, 85.0, 76.0, 58.0],
            "group": ["a", "a", "b", "b", "c", "c"],
        }
    )
    mfa = pd.DataFrame(
        {
            "length": rng.normal(size=12),
            "width": rng.normal(size=12),
            "color": np.resize(["red", "blue", "green"], 12),
            "shape": np.resize(["round", "square"], 12),
        }
    )
    rotations = prince.datasets.load_rotations().head(30).reset_index(drop=True)

    return [
        pytest.param(prince.PCA(n_components=2, engine="scipy"), ca, {}, id="pca"),
        pytest.param(prince.CA(n_components=2, engine="scipy"), ca, {}, id="ca"),
        pytest.param(prince.MCA(n_components=2, engine="scipy"), mca, {}, id="mca"),
        pytest.param(prince.FAMD(n_components=2, engine="scipy"), famd, {}, id="famd"),
        pytest.param(
            prince.MFA(n_components=2, engine="scipy"),
            mfa,
            {"groups": {"measurements": ["length", "width"], "traits": ["color", "shape"]}},
            id="mfa",
        ),
        pytest.param(prince.PGA(n_components=2, engine="scipy"), rotations, {}, id="pga"),
    ]


def _to_polars(X):
    return pl.DataFrame({column: X[column].to_numpy() for column in X.columns})


@pytest.mark.parametrize("estimator, X_pandas, fit_params", _cases())
@pytest.mark.parametrize(
    "implementation",
    [nw.Implementation.PANDAS, nw.Implementation.POLARS],
)
def test_dataframe_input_matches_pandas(estimator, X_pandas, fit_params, implementation):
    """Every estimator with two-dimensional input accepts pandas and Polars."""
    X = _to_polars(X_pandas) if implementation is nw.Implementation.POLARS else X_pandas

    expected_estimator = estimator.fit(X_pandas, **fit_params)
    actual_estimator = estimator.__class__(**estimator.get_params()).fit(X, **fit_params)
    method = "transform" if hasattr(estimator, "transform") else "row_coordinates"
    expected = getattr(expected_estimator, method)(X_pandas)
    actual = getattr(actual_estimator, method)(X)

    np.testing.assert_allclose(actual.abs(), expected.reset_index(drop=True).abs())
