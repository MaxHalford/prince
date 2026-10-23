from __future__ import annotations

import math

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
import rpy2.robjects as robjects
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import numpy2ri
from sklearn import decomposition, pipeline, preprocessing
from sklearn.exceptions import NotFittedError

import prince
from tests import assert_allclose_to_factominer, load_df_from_R


def as_container(X, container):
    if container is np.ndarray:
        return X.to_numpy()
    if container is nw.Implementation.POLARS:
        return pl.from_pandas(X)
    return X


@pytest.mark.parametrize(
    "container, sup_rows, sup_cols, scale, sample_weights, column_weights",
    [
        pytest.param(
            container,
            sup_rows,
            sup_cols,
            scale,
            sample_weights,
            column_weights,
            id=(
                f"container={container.__name__ if container is np.ndarray else container}:"
                f"{sup_rows=}:{sup_cols=}:{scale=}:{sample_weights=}:{column_weights=}"
            ),
        )
        for container in [
            nw.Implementation.PANDAS,
            np.ndarray,
            nw.Implementation.POLARS,
        ]
        for sup_rows in [False, True]
        for sup_cols in [False, True]
        for scale in [False, True]
        for sample_weights in [False, True]
        for column_weights in [False, True]
        if container is not np.ndarray or not sup_cols
    ],
)
class TestPCA:
    @pytest.fixture(autouse=True)
    def _prepare(self, container, sup_rows, sup_cols, scale, sample_weights, column_weights):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols
        self.scale = scale

        n_components = 5
        rng = np.random.default_rng(42)

        # Fit Prince
        self.dataset = prince.datasets.load_decathlon()
        self.active = self.dataset.copy()
        if self.sup_rows:
            self.active = self.active.query('competition == "Decastar"')
        self.dataset_input = as_container(self.dataset, container)
        self.active_input = as_container(self.active, container)
        self.sample_weights = rng.dirichlet([1] * len(self.active)) if sample_weights else None
        supplementary_columns = ["rank", "points"] if self.sup_cols else []
        self.active_columns = [
            column for column in self.active.columns if column not in supplementary_columns
        ]
        self.column_weights = rng.random(len(self.active_columns)) if column_weights else None
        self.pca = prince.PCA(n_components=n_components, rescale_with_std=self.scale)
        self.pca.fit(
            self.active_input,
            sample_weight=self.sample_weights,
            column_weight=self.column_weights,
            supplementary_columns=supplementary_columns,
        )

        # scikit-learn
        if self.scale:
            self.sk_pca = pipeline.make_pipeline(
                preprocessing.StandardScaler(),
                decomposition.PCA(n_components=n_components),
            )
        else:
            self.sk_pca = pipeline.make_pipeline(
                decomposition.PCA(n_components=n_components),
            )
        # sklearn's PCA doesn't support sample weights
        self.sk_pca.fit(self.active[self.active_columns])

        # Fit FactoMineR
        robjects.r(
            """
        library('FactoMineR')

        data(decathlon)
        decathlon <- subset(decathlon, select = -c(Competition))
        """
        )

        args = f"decathlon, ncp={n_components}, graph=F"
        if sample_weights:
            robjects.r.assign("row.w", numpy2ri.py2rpy(self.sample_weights))
            robjects.r("row.w <- as.vector(row.w)")
            args += ", row.w=row.w"
        if column_weights:
            robjects.r.assign("col.w", numpy2ri.py2rpy(self.column_weights))
            robjects.r("col.w <- as.vector(col.w)")
            args += ", col.w=col.w"
        if not self.scale:
            args += ", scale.unit=F"
        if self.sup_cols:
            if self.sup_rows:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12), ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args}, quanti.sup=c(11, 12))")
        else:
            if self.sup_rows:
                robjects.r(f"pca = PCA({args}, ind.sup=c(14:41))")
            else:
                robjects.r(f"pca = PCA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.pca, prince.PCA)
        sklearn.utils.validation.check_is_fitted(self.pca)

    def test_total_inertia(self):
        # FactoMineR 2.15 only stores the top `ncp` eigenvalues, so
        # `sum(pca$eig[,1])` no longer equals the true total inertia.
        # Recover it from the percentage of variance of the first component.
        F = robjects.r("pca$eig[1,1] / (pca$eig[1,2] / 100)")[0]
        P = self.pca.total_inertia_
        assert math.isclose(F, P)

    def test_eigenvalues(self):
        P = self.pca._eigenvalues_summary
        # Test against FactoMineR
        F = load_df_from_R("pca$eig")[: self.pca.n_components]
        assert_allclose_to_factominer(F["eigenvalue"], P["eigenvalue"])
        assert_allclose_to_factominer(F["percentage of variance"], P["% of variance"])
        assert_allclose_to_factominer(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )
        # Test against scikit-learn
        if self.sample_weights is None and self.column_weights is None:
            n = len(self.active)
            S = self.sk_pca[-1].explained_variance_ * (n - 1) / n
            np.testing.assert_allclose(P["eigenvalue"], S)
            np.testing.assert_allclose(
                P["% of variance"], self.sk_pca[-1].explained_variance_ratio_ * 100
            )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.pca, method_name)
        P = method(self.dataset_input)
        # Test againt FactoMineR
        F = load_df_from_R("pca$ind$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("pca$ind.sup$coord")))
        assert_allclose_to_factominer(F.abs(), P.abs())
        # Test against scikit-learn
        if self.sample_weights is None and self.column_weights is None:
            S = self.sk_pca.transform(self.dataset[self.active_columns])
            np.testing.assert_allclose(np.abs(S), P.abs())

    def test_row_cosine_similarities(self):
        F = load_df_from_R("pca$ind$cos2")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("pca$ind.sup$cos2")))
        P = self.pca.row_cosine_similarities(self.dataset_input)
        assert_allclose_to_factominer(F, P)

    def test_row_contrib(self):
        F = load_df_from_R("pca$ind$contrib")
        P = self.pca.row_contributions_
        assert_allclose_to_factominer(F, P * 100)

    def test_col_coords(self):
        F = load_df_from_R("pca$var$coord")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R("pca$quanti.sup$coord")))
        P = self.pca.column_coordinates_
        assert_allclose_to_factominer(F.abs(), P.abs())

    def test_col_cos2(self):
        F = load_df_from_R("pca$var$cos2")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R("pca$quanti.sup$cos2")))
        P = self.pca.column_cosine_similarities_
        assert_allclose_to_factominer(F, P)

    def test_col_contrib(self):
        F = load_df_from_R("pca$var$contrib")
        P = self.pca.column_contributions_
        # Contributions are percentages. Across BLAS implementations, the largest
        # observed relative difference is 5.42e-7, so keep this comparison bounded
        # to one part per million.
        assert_allclose_to_factominer(F, P * 100, rtol=1e-6)

    def test_col_cor(self):
        F = load_df_from_R("pca$var$cor")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R("pca$quanti.sup$cor")))
        P = self.pca.column_correlations
        assert_allclose_to_factominer(F.abs(), P.abs())


@pytest.mark.parametrize(
    "show_row_markers, show_row_labels, show_column_markers, show_column_labels",
    [
        (True, False, True, False),  # default biplot
        (True, False, False, False),  # rows only
        (False, False, True, False),  # columns only (used to raise on row scaling)
        (False, True, False, True),  # labels only, no markers
    ],
)
def test_plot_marker_combinations(
    show_row_markers, show_row_labels, show_column_markers, show_column_labels
):
    """`plot` should not require rows to be drawn in order to draw columns.

    The column coordinates are scaled into the row coordinate space, so the row
    coordinates must be available whenever the column chart is drawn — even when row
    markers and labels are both disabled.
    """
    pca = prince.PCA(n_components=2)
    pca.fit(prince.datasets.load_decathlon())
    chart = pca.plot(
        prince.datasets.load_decathlon(),
        show_row_markers=show_row_markers,
        show_row_labels=show_row_labels,
        show_column_markers=show_column_markers,
        show_column_labels=show_column_labels,
    )
    assert chart is not None


def test_get_feature_names_out():
    """get_feature_names_out returns one label per fitted component (sklearn transformer API)."""
    X = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4), columns=list("abcd"))
    pca = prince.PCA(n_components=3).fit(X)
    assert pca.get_feature_names_out().tolist() == ["pca0", "pca1", "pca2"]
    assert len(pca.get_feature_names_out()) == pca.transform(X).shape[1] == 3


def test_get_feature_names_out_checks_is_fitted():
    with pytest.raises(NotFittedError):
        prince.PCA().get_feature_names_out()


@pytest.mark.parametrize("weighted", [False, True])
def test_row_standard_coordinates_have_unit_variance(weighted):
    dataset = prince.datasets.load_decathlon()
    weights = np.arange(1, len(dataset) + 1, dtype=float) if weighted else np.ones(len(dataset))
    weights /= weights.sum()
    pca = prince.PCA(n_components=3, engine="scipy").fit(dataset, sample_weight=weights)
    coordinates = pca.row_standard_coordinates(dataset)
    np.testing.assert_allclose(
        coordinates.T @ (weights[:, None] * coordinates), np.eye(3), atol=1e-14
    )
    pd.testing.assert_index_equal(coordinates.index, dataset.index)


@pytest.mark.parametrize("size", [12, 41])
def test_refit_with_different_row_count(size):
    dataset = prince.datasets.load_decathlon()
    pca = prince.PCA(n_components=3, engine="scipy").fit(dataset.iloc[:20])
    current = dataset.iloc[:size]
    assert pca.fit(current) is pca
    fresh = prince.PCA(n_components=3, engine="scipy").fit(current)
    pd.testing.assert_frame_equal(pca.row_contributions_, fresh.row_contributions_)
    pd.testing.assert_frame_equal(pca.transform(current), fresh.transform(current))


@pytest.mark.parametrize(
    "fit_params",
    [
        pytest.param({"sample_weight": np.linspace(0.5, 2, 20)}, id="sample_weight"),
        pytest.param({"column_weight": np.array([1.0, 2.0, 0.5, 1.0])}, id="column_weight"),
        pytest.param({"supplementary_columns": ["d"]}, id="supplementary_columns"),
    ],
)
def test_fit_transform_uses_fit_params(fit_params):
    """fit_transform passes its fit parameters on to fit, like fit(X).transform(X)."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 4)), columns=list("abcd"))
    expected = prince.PCA(n_components=2).fit(X, **fit_params).transform(X)
    actual = prince.PCA(n_components=2).fit_transform(X, **fit_params)
    np.testing.assert_allclose(actual.abs(), expected.abs())


@sklearn.utils.estimator_checks.parametrize_with_checks([prince.PCA()])
def test_sklearn_estimator_compatibility(estimator, check):
    check(estimator)


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(np.ndarray, id="numpy"),
        pytest.param(nw.Implementation.POLARS, id="polars"),
    ],
)
def test_non_pandas_inputs_match_pandas(container):
    X_pandas = prince.datasets.load_decathlon()
    X = X_pandas.to_numpy() if container is np.ndarray else pl.from_pandas(X_pandas)

    expected = prince.PCA(n_components=3, engine="scipy").fit_transform(X_pandas)
    pca = prince.PCA(n_components=3, engine="scipy")
    actual = pca.fit_transform(X)

    assert isinstance(actual, pd.DataFrame)
    assert actual.index.equals(pd.RangeIndex(len(X_pandas)))
    np.testing.assert_allclose(actual.abs(), expected.reset_index(drop=True).abs())


def test_numpy_output():
    X = prince.datasets.load_decathlon().to_numpy()
    pca = prince.PCA(n_components=3, engine="scipy").fit(X)

    expected = pca.transform(X)
    actual = pca.transform(X, as_array=True)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected.to_numpy())


def test_polars_supplementary_columns_match_pandas():
    X_pandas = prince.datasets.load_decathlon()
    X_polars = pl.from_pandas(X_pandas)
    supplementary_columns = ["rank", "points"]

    expected = prince.PCA(n_components=3, engine="scipy").fit_transform(
        X_pandas,
        supplementary_columns=supplementary_columns,
    )
    pca = prince.PCA(n_components=3, engine="scipy")
    actual = pca.fit_transform(X_polars, supplementary_columns=supplementary_columns)

    assert pca.feature_names_in_.tolist() == [
        column for column in X_pandas.columns if column not in supplementary_columns
    ]
    np.testing.assert_allclose(actual.abs(), expected.reset_index(drop=True).abs())


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(np.ndarray, id="numpy"),
        pytest.param(nw.Implementation.PANDAS, id="pandas"),
        pytest.param(nw.Implementation.POLARS, id="polars"),
    ],
)
def test_set_output_polars(container):
    X_pandas = prince.datasets.load_decathlon()
    if container is np.ndarray:
        X = X_pandas.to_numpy()
    elif container is nw.Implementation.POLARS:
        X = pl.from_pandas(X_pandas)
    else:
        X = X_pandas

    result = (
        prince.PCA(n_components=3, engine="scipy").set_output(transform="polars").fit_transform(X)
    )

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["pca0", "pca1", "pca2"]


@pytest.mark.parametrize(
    "method_name",
    ["row_coordinates", "row_standard_coordinates", "row_cosine_similarities"],
)
def test_polars_row_methods_match_pandas(method_name):
    X_pandas = prince.datasets.load_decathlon()
    X_polars = pl.from_pandas(X_pandas)
    pca = prince.PCA(n_components=3, engine="scipy").fit(X_polars)

    expected = getattr(pca, method_name)(X_pandas)
    actual = getattr(pca, method_name)(X_polars)

    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))


def test_inverse_transform_accepts_polars():
    X = prince.datasets.load_decathlon()
    pca = prince.PCA(n_components=3, engine="scipy").fit(X)
    coordinates = pca.transform(X)

    expected = pca.inverse_transform(coordinates)
    actual = pca.inverse_transform(pl.from_pandas(coordinates))

    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))


def test_plot_accepts_polars():
    X = pl.from_pandas(prince.datasets.load_decathlon())
    pca = prince.PCA(n_components=3, engine="scipy").fit(X)

    assert pca.plot(X) is not None
