from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn import decomposition, preprocessing

import prince


def make_data():
    return pd.DataFrame(
        [[2.0, 7.0, 1.0], [4.0, 1.0, 5.0], [8.0, 3.0, 2.0], [3.0, 6.0, 9.0]],
        columns=["height", "width", "depth"],
        index=pd.Index(["b", "a", "b", "d"], name="sample"),
    )


@pytest.mark.parametrize("engine", ["scipy", "sklearn"])
@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("supplementary", [False, True])
def test_inverse_transform_preserves_active_columns(engine, n_components, supplementary):
    X = make_data()
    if supplementary:
        X.insert(1, "extra", [10.0, 20.0, 15.0, 30.0])
    pca = prince.PCA(n_components=n_components, engine=engine, random_state=42).fit(
        X, supplementary_columns=["extra"] if supplementary else None
    )
    scores = pca.transform(X.iloc[[3, 0, 2]])
    restored = pca.inverse_transform(scores)

    assert restored.columns.tolist() == ["height", "width", "depth"]
    pd.testing.assert_index_equal(restored.index, scores.index)
    pd.testing.assert_frame_equal(pca.transform(restored), scores, atol=1e-12)
    np.testing.assert_allclose(pca.inverse_transform(scores, as_array=True), restored)


def test_inverse_transform_supports_polars_frames():
    X_pandas = make_data()
    X_pandas.insert(1, "extra", [10.0, 20.0, 15.0, 30.0])
    X = pl.from_pandas(X_pandas)
    pca = prince.PCA(n_components=3, engine="scipy").set_output(transform="polars")
    pca.fit(X, supplementary_columns=["extra"])

    scores = pca.transform(X)
    restored = pca.inverse_transform(scores)

    assert isinstance(scores, pl.DataFrame)
    assert restored.columns.tolist() == ["height", "width", "depth"]
    np.testing.assert_allclose(restored, X.select("height", "width", "depth"), atol=1e-12)


def test_inverse_transform_keeps_integer_column_order():
    X = make_data().set_axis([20, 5, 11], axis="columns")
    pca = prince.PCA(n_components=3, engine="scipy").fit(X)
    restored = pca.inverse_transform(pca.transform(X))

    assert restored.columns.tolist() == [20, 5, 11]
    pd.testing.assert_frame_equal(restored, X, atol=1e-12)


@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("scale", [False, True])
def test_inverse_transform_array_input_matches_sklearn(n_components, scale):
    X = make_data().to_numpy()
    pca = prince.PCA(n_components=n_components, engine="scipy", rescale_with_std=scale)
    scores = pca.fit_transform(X)
    restored = pca.inverse_transform(scores.to_numpy())

    scaler = preprocessing.StandardScaler(with_std=scale)
    scaled = scaler.fit_transform(X)
    reference = decomposition.PCA(n_components=n_components, svd_solver="full")
    expected = scaler.inverse_transform(
        reference.inverse_transform(reference.fit_transform(scaled))
    )
    np.testing.assert_allclose(restored, expected, atol=1e-12)
    pd.testing.assert_index_equal(restored.columns, pd.RangeIndex(X.shape[1]))
    pd.testing.assert_index_equal(restored.index, pd.RangeIndex(X.shape[0]))
    np.testing.assert_allclose(pca.transform(restored.to_numpy()), scores, atol=1e-12)


def test_inverse_transform_after_refit_uses_current_columns():
    X = make_data()
    pca = prince.PCA(n_components=2, engine="scipy").fit(X)
    current = X.rename(columns={"height": "length", "width": "breadth", "depth": "thickness"})
    scores = pca.fit_transform(current)
    assert pca.inverse_transform(scores).columns.tolist() == current.columns.tolist()

    scores = pca.fit_transform(X.to_numpy())
    assert pca.inverse_transform(scores).columns.tolist() == [0, 1, 2]


def test_inverse_transform_with_numpy_scores_keeps_fitted_columns():
    X = make_data()
    pca = prince.PCA(n_components=3, engine="scipy").fit(X)
    scores = pca.transform(X, as_array=True)
    restored = pca.inverse_transform(scores)

    assert restored.columns.tolist() == X.columns.tolist()
    np.testing.assert_allclose(restored, X, atol=1e-12)
    pd.testing.assert_index_equal(restored.index, pd.RangeIndex(len(X)))
