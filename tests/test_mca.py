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
