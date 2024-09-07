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

        args = f"dataset, ncp={n_components}, graph=F"
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

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        super().test_row_coords(method_name=method_name)

    def test_col_coords(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$coord")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$coord")))
            P = self.ca.column_coordinates(self.dataset)
            np.testing.assert_allclose(F.abs(), P.abs())
        else:
            super().test_col_coords()

    def test_col_cos2(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$cos2")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$cos2")))
            P = self.ca.column_cosine_similarities(self.dataset)
            np.testing.assert_allclose(F, P)
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
    >>> mca.transform(df).round(2).abs().sort_index(axis='columns')
         0     1
    0  2.0  0.00
    1  0.5  0.65
    2  0.5  0.65
    3  0.5  0.65
    4  0.5  1.94

    >>> mca = prince.MCA(n_components=2, one_hot=False, engine="scipy")
    >>> one_hot = pd.get_dummies(df, columns=['foo', 'bar'])
    >>> mca = mca.fit(one_hot)
    >>> mca.transform(one_hot).round(2).abs().sort_index(axis='columns')
         0     1
    0  2.0  0.00
    1  0.5  0.65
    2  0.5  0.65
    3  0.5  0.65
    4  0.5  1.94

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
    >>> mca.transform(df).round(2).abs().sort_index(axis='columns')
         0     1
    0  2.0  0.00
    1  0.5  0.65
    2  0.5  0.65
    3  0.5  0.65
    4  0.5  1.94

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
