from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r as R

import prince
from tests import load_df_from_R
from tests.test_ca import TestCA as _TestCA


class TestMCA(_TestCA):
    _row_name = "ind"
    _col_name = "var"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5
        n_active_rows = 1_000
        # FactoMineR 2.15's `svd.triplet` uses `irlba` whenever
        # `ncp < 0.5 * min(nrow_active, ncol_indicator)`. For the Hearthstone
        # dataset that means anything below ~19 picks the iterative solver,
        # whose approximation on later components drifts well past Prince's
        # full-SVD precision (1-5% relative error on component 5). Asking
        # FactoMineR for more components forces the full-SVD branch; we then
        # slice the outputs back down to `n_components` before comparing.
        n_components_R = 20

        # Fit Prince
        self.dataset = prince.datasets.load_hearthstone_cards()
        active = self.dataset.copy()
        if self.sup_rows:
            active = active[:n_active_rows]
        if self.sup_cols:
            active = active.drop(columns=["type_or_school"])
        self.ca = prince.MCA(n_components=n_components, engine="scipy")
        self.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            self.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}')[,-1]")

        args = f"dataset, ncp={n_components_R}, graph=F"
        if self.sup_cols:
            if self.sup_rows:
                R(
                    f"ca <- MCA({args}, quali.sup=c(4), ind.sup=c({n_active_rows + 1}:nrow(dataset)))"
                )
            else:
                R(f"ca <- MCA({args}, quali.sup=c(4))")
        else:
            if self.sup_rows:
                R(f"ca <- MCA({args}, ind.sup=c({n_active_rows + 1}:nrow(dataset)))")
            else:
                R(f"ca <- MCA({args})")

        # Slice FactoMineR's outputs back down to `n_components` so the rest
        # of the assertions (inherited from TestCA) compare matching shapes.
        R(f"""
        nc <- {n_components}
        ca$svd$U <- ca$svd$U[, 1:nc, drop=FALSE]
        ca$svd$V <- ca$svd$V[, 1:nc, drop=FALSE]
        ca$ind$coord   <- ca$ind$coord[, 1:nc, drop=FALSE]
        ca$ind$cos2    <- ca$ind$cos2[, 1:nc, drop=FALSE]
        ca$ind$contrib <- ca$ind$contrib[, 1:nc, drop=FALSE]
        ca$var$coord   <- ca$var$coord[, 1:nc, drop=FALSE]
        ca$var$cos2    <- ca$var$cos2[, 1:nc, drop=FALSE]
        ca$var$contrib <- ca$var$contrib[, 1:nc, drop=FALSE]
        if (!is.null(ca$ind.sup))   ca$ind.sup$coord    <- ca$ind.sup$coord[, 1:nc, drop=FALSE]
        if (!is.null(ca$ind.sup))   ca$ind.sup$cos2     <- ca$ind.sup$cos2[, 1:nc, drop=FALSE]
        if (!is.null(ca$quali.sup)) ca$quali.sup$coord  <- ca$quali.sup$coord[, 1:nc, drop=FALSE]
        if (!is.null(ca$quali.sup)) ca$quali.sup$cos2   <- ca$quali.sup$cos2[, 1:nc, drop=FALSE]
        """)

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        super().test_row_coords(method_name=method_name)

    def test_col_coords(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$coord")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$coord")))
            P = self.ca.column_coordinates(self.dataset)
            # Prince adds a prefix to each column. We need to remove it in order to align the rows
            # of the two dataframes
            P.index = [idx.split("__", 1)[1] for idx in P.index]
            np.testing.assert_allclose(F.abs(), P.abs().loc[F.index])
        else:
            super().test_col_coords()

    def test_col_cos2(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$cos2")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$cos2")))
            P = self.ca.column_cosine_similarities(self.dataset)
            # Prince adds a prefix to each column. We need to remove it in order to align the rows
            # of the two dataframes
            P.index = [idx.split("__", 1)[1] for idx in P.index]
            np.testing.assert_allclose(F, P.loc[F.index])
        else:
            super().test_col_cos2()


def test_with_and_without_one_hot():
    """

    >>> df = pd.DataFrame({
    ...     "foo": [1, 2, 3, 3, 5],
    ...     "bar": ["a", "b", "c", "b", "e"],
    ... })
    >>> mca = prince.MCA(n_components=2, one_hot=True, engine="scipy")
    >>> mca = mca.fit(df)
    >>> coords = mca.transform(df)
    >>> assert coords.shape == (5, 2)
    >>> coords.round(2).abs().sort_index(axis='columns')  # doctest: +SKIP
          0    1
    0  0.00  2.0
    1  0.65  0.5
    2  0.65  0.5
    3  0.65  0.5
    4  1.94  0.5

    >>> mca = prince.MCA(n_components=2, one_hot=False, engine="scipy")
    >>> one_hot = pd.get_dummies(df, columns=['foo', 'bar'])
    >>> mca = mca.fit(one_hot)
    >>> coords = mca.transform(one_hot)
    >>> assert coords.shape == (5, 2)
    >>> coords.round(2).abs().sort_index(axis='columns')  # doctest: +SKIP
          0    1
    0  0.00  1.0
    1  0.65  0.5
    2  0.65  0.5
    3  0.65  0.5
    4  1.94  0.5

    """


def test_issue_131():
    """

    https://github.com/MaxHalford/prince/issues/131#issuecomment-1591426031

    >>> df = pd.DataFrame({
    ...     "foo": [1, 2, 3, 3, 5],
    ...     "bar": ["a", "b", "c", "b", "e"],
    ... })
    >>> mca = prince.MCA(engine="scipy")
    >>> mca = mca.fit(df)
    >>> coords = mca.transform(df)
    >>> assert coords.shape == (5, 2)
    >>> coords.round(2).abs().sort_index(axis='columns')  # doctest: +SKIP
          0    1
    0  0.00  2.0
    1  0.65  0.5
    2  0.65  0.5
    3  0.65  0.5
    4  1.94  0.5

    >>> mca.K_, mca.J_
    (2, 8)

    """


def test_issue_171():
    """

    https://github.com/MaxHalford/prince/issues/171

    >>> from sklearn import impute
    >>> from sklearn import pipeline

    >>> rng = np.random.RandomState(0)
    >>> test_data = pd.DataFrame(data=rng.random((10, 5)))
    >>> test = pipeline.Pipeline(steps=[
    ...     ('impute', impute.SimpleImputer()),  # would break the pipeline since it returns an ndarray
    ...     ('mca', prince.PCA()),
    ... ])
    >>> _ = test[0].set_output(transform='pandas')
    >>> test.fit_transform(test_data)
    component         0         1
    0         -0.392617  0.296831
    1          0.119661 -1.660653
    2         -1.541581 -0.826863
    3          3.105498 -0.538801
    4         -2.439259 -0.343292
    5          1.129341 -0.533576
    6         -1.077436  0.899673
    7          0.020571 -0.941029
    8          1.498005  1.566376
    9         -0.422184  2.081334

    """


def test_type_doesnt_matter():
    """

    Checks that the type of the columns doesn't affect the result.

    """
    outputs = []
    dataset = prince.datasets.load_hearthstone_cards().head(100)
    for col in dataset.columns:
        labels, levels = pd.factorize(dataset[col])
        dataset[col] = labels
    for typ in ("int", "float", "str", "category"):
        dataset = dataset.astype(typ)
        mca = prince.MCA(n_components=2, engine="scipy")
        mca = mca.fit(dataset)
        outputs.append(mca.transform(dataset).abs())

    for i in range(len(outputs) - 1):
        np.testing.assert_allclose(outputs[i], outputs[i + 1])


issue_161_data = """
,category,userid,location,applicationname,browser\n
0,Portal Login,a@b.com,"San Jose, CA, United States",A,Chrome\n
1,Application Access,b@b.com,"San Jose, CA, United States",B,Other\n
2,Application Access,a@b.com,"San Jose, CA, United States",C,Other\n
3,Portal Login,c@b.com,"San Diego, CA, United States",A,Chrome\n
"""


def test_issue_161():
    """

    https://github.com/MaxHalford/prince/issues/161

    >>> import io
    >>> data = pd.read_csv(io.StringIO(issue_161_data), index_col=0)

    >>> mca = prince.MCA(
    ...     n_components=10,
    ...     n_iter=3,
    ...     copy=True,
    ...     check_input=True,
    ...     engine='sklearn',
    ...     random_state=42
    ... )
    >>> mca = mca.fit(data[:3])

    >>> mca.eigenvalues_summary
              eigenvalue % of variance % of variance (cumulative)
    component
    0              0.673        67.32%                     67.32%
    1              0.327        32.68%                    100.00%

    >>> mca.row_coordinates(data[:3])
              0         1
    0  1.120811 -0.209242
    1 -0.820491 -0.571660
    2 -0.300320  0.780902

    >>> mca.transform(data[3:])
              0         1
    3  1.664888 -0.640285

    """


def test_non_subset_correction_matches_ca_mjca():
    """Non-subset Benzécri/Greenacre corrections match R's ``ca::mjca(lambda='adjusted')``.

    FactoMineR doesn't expose these corrections, but Greenacre's own ``ca`` package does,
    and its closed-form is what prince's non-subset path implements.
    """
    wines = prince.datasets.load_burgundy_wines().drop(columns=["Oak type"], level=0)
    wines.columns = [f"{a}_{b}" for a, b in wines.columns]

    R("library('ca')")
    with tempfile.NamedTemporaryFile(suffix=".csv") as fp:
        wines.to_csv(fp.name, index=False)
        R(f"dataset <- read.csv('{fp.name}')")
    R("""
    dataset <- data.frame(lapply(dataset, factor))
    mj <- mjca(dataset, lambda='adjusted', nd=4)
    """)
    r_lambda = np.array(R("mj$sv")) ** 2
    r_inertia_e = np.array(R("mj$inertia.e"))
    n_nonzero = int(np.sum(r_lambda > 0))

    mca_g = prince.MCA(n_components=4, correction="greenacre").fit(wines)
    np.testing.assert_allclose(mca_g.eigenvalues_[:n_nonzero], r_lambda[:n_nonzero], atol=1e-8)
    np.testing.assert_allclose(
        mca_g.percentage_of_variance_[:n_nonzero] / 100, r_inertia_e[:n_nonzero], atol=1e-8
    )

    # Benzécri shares the same adjusted eigenvalues; only the percentages differ — they
    # renormalise to sum to 100% instead of using Greenacre's adjusted-inertia denominator.
    mca_b = prince.MCA(n_components=4, correction="benzecri").fit(wines)
    np.testing.assert_allclose(mca_b.eigenvalues_[:n_nonzero], r_lambda[:n_nonzero], atol=1e-8)
    expected_benzecri_pct = r_lambda[:n_nonzero] / r_lambda.sum() * 100
    np.testing.assert_allclose(
        mca_b.percentage_of_variance_[:n_nonzero], expected_benzecri_pct, atol=1e-8
    )


def test_subset_greenacre_matches_ca_mjca():
    """Subset-MCA Greenacre correction matches R's ``ca::mjca(lambda='adjusted', subsetcat=...)``.

    Background: https://github.com/MaxHalford/prince/issues/206
    """
    wines = prince.datasets.load_burgundy_wines().drop(columns=["Oak type"], level=0)
    wines.columns = [f"{a}_{b}" for a, b in wines.columns]
    sep = "__"
    one_hot = pd.get_dummies(wines, columns=wines.columns, prefix_sep=sep)
    to_drop = [c for c in one_hot.columns if c.endswith(f"{sep}No")]

    mca = prince.MCA(
        n_components=4,
        correction="greenacre",
        one_hot_prefix_sep=sep,
        one_hot_columns_to_drop=to_drop,
    ).fit(wines)

    R("library('ca')")
    with tempfile.NamedTemporaryFile(suffix=".csv") as fp:
        wines.to_csv(fp.name, index=False)
        R(f"dataset <- read.csv('{fp.name}')")
    R("""
    dataset <- data.frame(lapply(dataset, factor))
    lvl_lens <- sapply(dataset, nlevels)
    offs <- c(0, cumsum(lvl_lens)[-length(lvl_lens)])
    subsetcat <- c()
    for (i in seq_along(dataset)) {
      lv <- levels(dataset[[i]])
      for (j in seq_along(lv)) {
        if (lv[j] != 'No') subsetcat <- c(subsetcat, offs[i] + j)
      }
    }
    mj <- mjca(dataset, lambda='adjusted', subsetcat=subsetcat, nd=4)
    """)
    r_lambda = np.array(R("mj$sv"))[: mca.eigenvalues_.shape[0]] ** 2
    r_inertia_e = np.array(R("mj$inertia.e"))
    r_inertia_t = float(np.array(R("mj$inertia.t"))[0])

    np.testing.assert_allclose(mca.eigenvalues_[: len(r_lambda)], r_lambda, atol=1e-8)
    np.testing.assert_allclose(
        mca.percentage_of_variance_[: len(r_inertia_e)] / 100, r_inertia_e, atol=1e-8
    )
    np.testing.assert_allclose(
        mca.percentage_of_variance_[: len(r_inertia_e)] / 100,
        mca.eigenvalues_[: len(r_inertia_e)] / r_inertia_t,
        atol=1e-10,
    )


def test_abdi_2007_correction():
    """

    >>> wines = prince.datasets.load_burgundy_wines()
    >>> wines = wines.drop(columns=["Oak type"], level=0)

    >>> mca = prince.MCA(n_components=4, correction=None)
    >>> mca = mca.fit(wines)
    >>> mca.eigenvalues_.round(4).tolist()
    [0.8532, 0.2, 0.1151, 0.0317]
    >>> mca.percentage_of_variance_.round(3).tolist()
    [71.101, 16.667, 9.593, 2.64]

    >>> mca = prince.MCA(n_components=4, correction="benzecri")
    >>> mca = mca.fit(wines)
    >>> mca.eigenvalues_.round(4).tolist()
    [0.7004, 0.0123, 0.0003, 0.0]
    >>> mca.percentage_of_variance_.round(3).tolist()
    [98.229, 1.731, 0.04, 0.0]

    >>> mca = prince.MCA(n_components=4, correction="greenacre")
    >>> mca = mca.fit(wines)
    >>> mca.eigenvalues_.round(4).tolist()
    [0.7004, 0.0123, 0.0003, 0.0]
    >>> mca.percentage_of_variance_.round(3).tolist()
    [95.189, 1.678, 0.038, 0.0]

    """
