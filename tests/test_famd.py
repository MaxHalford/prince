from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import r as R

import prince
from tests import load_df_from_R


@pytest.mark.parametrize(
    "sup_rows, sup_cols",
    [
        pytest.param(
            sup_rows,
            sup_cols,
            id=":".join(["sup_rows" if sup_rows else "", "sup_cols" if sup_cols else ""]).strip(
                ":"
            ),
        )
        for sup_rows in [False]
        for sup_cols in [False]
    ],
)
class TestFAMD:
    _row_name = "row"
    _col_name = "col"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):
        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5

        # Fit Prince
        self.dataset = prince.datasets.load_beers().head(200)
        active = self.dataset.copy()
        self.famd = prince.FAMD(n_components=n_components, engine="scipy")
        self.famd.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            self.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=c(1))")
            R("famd <- FAMD(dataset, graph=F)")

    def test_check_is_fitted(self):
        assert isinstance(self.famd, prince.FAMD)
        sklearn.utils.validation.check_is_fitted(self.famd)

    def test_num_cols(self):
        assert sorted(self.famd.num_cols_) == [
            "alcohol_by_volume",
            "final_gravity",
            "international_bitterness_units",
            "standard_reference_method",
        ]

    def test_cat_cols(self):
        assert sorted(self.famd.cat_cols_) == ["is_organic", "style"]

    def test_eigenvalues(self):
        F = load_df_from_R("famd$eig")[: self.famd.n_components]
        P = self.famd._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.famd, method_name)
        F = load_df_from_R("famd$ind$coord")
        P = method(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R("famd$ind$contrib")
        P = self.famd.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_variable_coords(self):
        # variable_coordinates_ matches FactoMineR's famd$var$coord directly:
        # r² for numerical, η² for categorical (one row per original variable).
        F = load_df_from_R("famd$var$coord")
        np.testing.assert_allclose(F, self.famd.variable_coordinates_, atol=1e-8)

    def test_variable_contrib(self):
        F = load_df_from_R("famd$var$contrib")
        np.testing.assert_allclose(F, self.famd.variable_contributions_ * 100, atol=1e-8)

    def test_col_coords_modality_level(self):
        # column_coordinates_ is at the preprocessed-column level (one row per
        # numerical + one row per modality), like MCA. Sanity check: squaring the
        # numerical rows reproduces the variable-level r² entries.
        cc = self.famd.column_coordinates_
        for col in self.famd.num_cols_:
            np.testing.assert_allclose(
                cc.loc[col] ** 2,
                self.famd.variable_coordinates_.loc[col],
                atol=1e-10,
            )
        # And summing squared modality coords reproduces the η² entries.
        for col in self.famd.cat_cols_:
            mods = [m for m in self.famd.one_hot_columns_ if m.startswith(f"{col}_")]
            np.testing.assert_allclose(
                (cc.loc[mods] ** 2).sum(axis=0),
                self.famd.variable_coordinates_.loc[col],
                atol=1e-10,
            )

    # -- Comparison with FactoMineR's per-type outputs --

    def test_quanti_var_coord_matches_factominer(self):
        """Numerical rows of column_coordinates_ = FactoMineR's famd$quanti.var$coord
        (signed Pearson correlations)."""
        F = load_df_from_R("famd$quanti.var$coord")
        P = self.famd.column_coordinates_.loc[self.famd.num_cols_]
        np.testing.assert_allclose(F.abs().values, P.abs().values, atol=1e-8)

    def test_quanti_var_contrib_matches_factominer(self):
        """Numerical rows of column_contributions_ * 100 = famd$quanti.var$contrib."""
        F = load_df_from_R("famd$quanti.var$contrib")
        P = self.famd.column_contributions_.loc[self.famd.num_cols_] * 100
        np.testing.assert_allclose(F.values, P.values, atol=1e-8)

    def _prince_modality(self, factominer_label):
        """Map FactoMineR's bare-category label (e.g. ``"Altbier"``) to prince's
        prefixed modality name (e.g. ``"style_Altbier"``). FactoMineR strips the
        column prefix; prince keeps ``{col}_{cat}``."""
        for col in self.famd.cat_cols_:
            cats = {str(c) for c in self.famd.categories_[col]}
            if factominer_label in cats:
                return f"{col}_{factominer_label}"
        raise KeyError(factominer_label)

    def test_quali_var_coord_matches_factominer_via_pages_scaling(self):
        """Modality rows of column_coordinates_ are Pagès's G_s(k_q) (PCA coord on the
        MCA-coded indicator). FactoMineR's famd$quali.var$coord stores the barycentres
        F_s(k_q). They're related by F_s = G_s · √(λ/p) (Pagès 2004 §5.1).

        FactoMineR orders modalities differently than pandas (locale-aware on the cat
        values), so we align by reindexing prince's output to FactoMineR's order.
        """
        F = load_df_from_R("famd$quali.var$coord")
        prince_mods = [self._prince_modality(m) for m in F.index]
        G = self.famd.column_coordinates_.loc[prince_mods].to_numpy()

        n = len(self.dataset)
        p_by_mod = {
            f"{col}_{cat}": (self.dataset[col].astype(str) == str(cat)).sum() / n
            for col in self.famd.cat_cols_
            for cat in self.famd.categories_[col]
        }
        p = np.array([p_by_mod[m] for m in prince_mods])
        lam = self.famd.eigenvalues_
        expected_bary = G * (np.sqrt(lam) / np.sqrt(p)[:, None])
        np.testing.assert_allclose(np.abs(F.values), np.abs(expected_bary), atol=1e-8)

    def test_quali_var_contrib_matches_factominer(self):
        """Modality rows of column_contributions_ * 100 = famd$quali.var$contrib
        (after aligning modality order)."""
        F = load_df_from_R("famd$quali.var$contrib")
        prince_mods = [self._prince_modality(m) for m in F.index]
        P = self.famd.column_contributions_.loc[prince_mods] * 100
        np.testing.assert_allclose(F.values, P.values, atol=1e-8)


class TestWikipediaExample:
    """Reproduce the example from the Wikipedia FAMD article.

    https://en.wikipedia.org/wiki/Factor_analysis_of_mixed_data

    Reference values computed with FactoMineR 2.11.
    """

    @pytest.fixture(autouse=True)
    def _prepare(self):
        import pandas as pd

        self.dataset = pd.DataFrame(
            {
                "k1": [2.0, 5.0, 3.0, 4.0, 1.0, 6.0],
                "k2": [4.5, 4.5, 1.0, 1.0, 1.0, 1.0],
                "k3": [4.0, 4.0, 2.0, 2.0, 1.0, 2.0],
                "q1": ["A", "C", "B", "B", "A", "C"],
                "q2": ["B", "B", "B", "B", "A", "A"],
                "q3": ["C", "C", "B", "B", "A", "A"],
            },
            index=[f"i{i}" for i in range(1, 7)],
        )
        self.num_cols = ["k1", "k2", "k3"]
        self.cat_cols = ["q1", "q2", "q3"]
        self.famd = prince.FAMD(n_components=5, engine="scipy")
        self.famd.fit(self.dataset)

    def test_eigenvalues(self):
        expected = np.array([3.466804028, 2.500000000, 1.971104825, 0.055852845, 0.006238302])
        np.testing.assert_allclose(self.famd.eigenvalues_, expected)

    # -- Table 2: Relationship matrix --

    def test_relationship_matrix(self):
        """Table 2 from Wikipedia: pairwise associations between all variables.

        R² for quanti-quanti, η² for quanti-quali, ϕ² (= χ²/n) for quali-quali.
        """
        from scipy.stats import chi2_contingency

        df = self.dataset
        all_cols = self.num_cols + self.cat_cols

        # fmt: off
        expected = np.array([
            [1.00, 0.00, 0.05, 0.91, 0.00, 0.00],
            [0.00, 1.00, 0.90, 0.25, 0.25, 1.00],
            [0.05, 0.90, 1.00, 0.13, 0.40, 0.93],
            [0.91, 0.25, 0.13, 2.00, 0.25, 1.00],
            [0.00, 0.25, 0.40, 0.25, 1.00, 1.00],
            [0.00, 1.00, 0.93, 1.00, 1.00, 2.00],
        ])
        # fmt: on

        R = np.zeros((6, 6))
        for i, ci in enumerate(all_cols):
            for j, cj in enumerate(all_cols):
                ci_num = ci in self.num_cols
                cj_num = cj in self.num_cols
                if ci_num and cj_num:
                    # R²
                    R[i, j] = df[ci].corr(df[cj]) ** 2
                elif ci_num != cj_num:
                    # η²
                    quanti, quali = (ci, cj) if ci_num else (cj, ci)
                    grand_mean = df[quanti].mean()
                    groups = df.groupby(quali)[quanti]
                    ss_b = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
                    ss_t = ((df[quanti] - grand_mean) ** 2).sum()
                    R[i, j] = ss_b / ss_t
                else:
                    # ϕ² = χ²/n
                    import pandas as pd

                    tab = pd.crosstab(df[ci], df[cj])
                    chi2 = chi2_contingency(tab, correction=False)[0]
                    R[i, j] = chi2 / len(df)

        np.testing.assert_allclose(R, expected, atol=0.01)

    # -- Figure 1: Individuals --

    def test_row_coordinates(self):
        """Figure 1: individual factor map.

        FactoMineR: famd$ind$coord (signs may differ per component).
        """
        expected = np.array(
            [
                [2.4125225, 0.4225771, 1.7279514, 0.2924097, 0.03567833],
                [2.7075166, 0.4225771, 1.2081045, 0.3178746, 0.04061852],
                [0.8748179, 2.1128856, 0.1178896, 0.1629730, 0.12515600],
                [0.8318618, 2.1128856, 0.2911719, 0.1714613, 0.12350927],
                [2.1351079, 1.6903085, 1.7270550, 0.2065304, 0.04067948],
                [1.2782515, 1.6903085, 2.0736196, 0.2235071, 0.04397294],
            ]
        )
        P = self.famd.row_coordinates(self.dataset)
        np.testing.assert_allclose(P.abs().values, expected, atol=1e-5)

    def test_row_contributions(self):
        # FactoMineR: famd$ind$contrib (as percentages)
        expected = np.array(
            [
                [27.980933, 1.190476, 25.246553, 25.514490, 3.400882],
                [35.242096, 1.190476, 12.340935, 30.151938, 4.407888],
                [3.679212, 29.761905, 0.117514, 7.925643, 41.849059],
                [3.326762, 29.761905, 0.716866, 8.772746, 40.755054],
                [21.915889, 19.047619, 25.220365, 12.728330, 4.421129],
                [7.855107, 19.047619, 36.357767, 14.906852, 5.165988],
            ]
        )
        np.testing.assert_allclose(self.famd.row_contributions_.values * 100, expected, atol=1e-4)

    # -- Figure 2: Relationship square --

    def test_column_coordinates(self):
        """Figure 2: relationship square (r² for quanti, η² for quali).

        FactoMineR's famd$var$coord is the variable-level inertia matrix. With the new
        FAMD API this lives in ``variable_coordinates_`` directly.
        """
        expected = np.array(
            [
                [0.01865808, 0.00000000, 0.96199662, 0.01822022, 0.00112508],
                [0.94520776, 0.03571429, 0.01713765, 0.00145128, 0.00048902],
                [0.98166131, 0.00000000, 0.00532657, 0.00963305, 0.00337907],
                [0.12096908, 0.89285714, 0.96125488, 0.02439825, 0.00052065],
                [0.42009234, 0.57142857, 0.00761673, 0.00064501, 0.00021734],
                [0.98021546, 1.00000000, 0.01777237, 0.00150503, 0.00050714],
            ]
        )
        np.testing.assert_allclose(self.famd.variable_coordinates_.values, expected, atol=1e-5)

    def test_column_contributions(self):
        # FactoMineR's famd$var$contrib (as percentages).
        expected = np.array(
            [
                [0.538193, 0.000000, 48.804945, 32.621824, 18.035038],
                [27.264528, 1.428571, 0.869444, 2.598402, 7.839055],
                [28.316031, 0.000000, 0.270233, 17.247195, 54.166541],
                [3.489355, 35.714286, 48.767314, 43.683095, 8.345951],
                [12.117568, 22.857143, 0.386419, 1.154845, 3.484024],
                [28.274326, 40.000000, 0.901645, 2.694639, 8.129390],
            ]
        )
        np.testing.assert_allclose(
            self.famd.variable_contributions_.values * 100, expected, atol=1e-4
        )

    def test_column_contributions_sum_to_one(self):
        # Per-preprocessed-column contributions sum to 1 per component.
        sums = self.famd.column_contributions_.sum()
        np.testing.assert_allclose(sums.values, 1.0)
        # Aggregated variable-level contributions also sum to 1 per component.
        np.testing.assert_allclose(self.famd.variable_contributions_.sum().values, 1.0)

    # -- Figure 3: Correlation circle --

    def test_correlation_circle(self):
        """Figure 3: correlation circle for quantitative variables.

        FactoMineR: famd$quanti.var$coord
        Signed correlations between standardized quantitative variables and components.
        Exposed via column_correlations property.
        """
        expected_abs = np.array(
            [
                [0.1365946, 0.0000000, 0.9808143, 0.1349823, 0.0335422],
                [0.9722180, 0.1889822, 0.1309108, 0.0380957, 0.0221139],
                [0.9907882, 0.0000000, 0.0729834, 0.0981481, 0.0581298],
            ]
        )
        cc = self.famd.column_correlations
        # Quantitative: signed correlations match FactoMineR quanti.var$coord
        np.testing.assert_allclose(cc.loc[self.num_cols].abs().values, expected_abs, atol=1e-5)
        # For numerical variables, column_coordinates_ now stores the signed correlations
        # directly (genuine PCA coords on standardized input), so the two are equal.
        np.testing.assert_allclose(
            self.famd.column_coordinates_.loc[self.num_cols].values,
            cc.loc[self.num_cols].values,
            atol=1e-5,
        )

    # -- Figure 4: Category representation --

    def test_category_coordinates(self):
        """Figure 4: category-level coordinates (barycentres of row coordinates).

        FactoMineR: famd$quali.var$coord
        Each category's coordinate is the mean of row coordinates for individuals
        belonging to that category.
        """
        # fmt: off
        expected_abs = np.array([
            # q1
            [0.1387073, 1.0564428, 1.7275032, 0.0429396, 0.0025006],  # A
            [0.8533399, 2.1128856, 0.0866412, 0.0042442, 0.0008234],  # B
            [0.7146326, 1.0564428, 1.6408621, 0.0471838, 0.0016772],  # C
            # q2
            [1.7066797, 1.6903085, 0.1732823, 0.0084883, 0.0016467],  # A
            [0.8533399, 0.8451543, 0.0866412, 0.0042442, 0.0008234],  # B
            # q3
            [1.7066797, 1.6903085, 0.1732823, 0.0084883, 0.0016467],  # A
            [0.8533399, 2.1128856, 0.0866412, 0.0042442, 0.0008234],  # B
            [2.5600196, 0.4225771, 0.2599234, 0.0127325, 0.0024701],  # C
        ])
        # fmt: on

        rc = self.famd.row_coordinates(self.dataset)
        rows = []
        for q in self.cat_cols:
            for cat in sorted(self.dataset[q].unique()):
                rows.append(rc[self.dataset[q] == cat].mean().values)
        actual = np.array(rows)
        np.testing.assert_allclose(np.abs(actual), expected_abs, atol=1e-5)


def test_issue_169():
    """

    https://github.com/MaxHalford/prince/issues/169

    >>> import pandas as pd
    >>> from prince import FAMD
    >>> df = pd.DataFrame({'var1':['c', 'a', 'b','c'], 'var2':['x','y','y','z'],'var2': [0.,10.,30.4,0.]})

    >>> famd = FAMD(n_components=2, random_state=42)
    >>> famd = famd.fit(df[:3])

    >>> famd.transform(df[0:3])
    component         0         1
    0         -1.505452 -0.931025
    1         -0.387542  1.387410
    2          1.892994 -0.456385

    >>> famd.transform(df[0:2])
    component         0         1
    0         -0.980918 -0.946497
    1          0.034421  0.946497

    >>> famd.transform(df[3:]).round(6)
    component         0    1
    3         -0.752726  0.0

    """


def test_categorical_columns_explicit():
    """User can override dtype-based auto-detection by passing ``categorical_columns``.

    Without ``categorical_columns``, only float columns are treated as numerical. An
    integer column like ``rating`` would be detected as categorical. Passing
    ``categorical_columns`` is the escape hatch.
    """
    import prince
    import pandas as pd

    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [4.5, 4.5, 1.0, 1.0, 1.0, 1.0],
            "rating": [1, 2, 1, 2, 3, 3],  # int — auto-detected as categorical
            "label": ["A", "B", "B", "B", "A", "A"],
        }
    )
    famd = prince.FAMD(n_components=2, engine="scipy").fit(df)
    assert sorted(famd.num_cols_) == ["x1", "x2"]
    assert sorted(famd.cat_cols_) == ["label", "rating"]

    # Override: treat rating as numerical.
    famd = prince.FAMD(n_components=2, engine="scipy", categorical_columns=["label"]).fit(df)
    assert sorted(famd.num_cols_) == ["rating", "x1", "x2"]
    assert sorted(famd.cat_cols_) == ["label"]


def test_supplementary_columns():
    """Supplementary columns are projected onto the active factor space without
    influencing how the axes are computed."""
    import prince
    import pandas as pd

    df = pd.DataFrame(
        {
            "x1": [2.0, 5.0, 3.0, 4.0, 1.0, 6.0],
            "x2": [4.5, 4.5, 1.0, 1.0, 1.0, 1.0],
            "x3": [4.0, 4.0, 2.0, 2.0, 1.0, 2.0],
            "q1": ["A", "C", "B", "B", "A", "C"],
            "q2": ["B", "B", "B", "B", "A", "A"],
            "q3_sup": ["C", "C", "B", "B", "A", "A"],
            "x_sup": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        },
        index=[f"i{i}" for i in range(1, 7)],
    )

    famd_full = prince.FAMD(n_components=4, engine="scipy").fit(df)
    famd_sup = prince.FAMD(n_components=4, engine="scipy").fit(
        df, supplementary_columns=["q3_sup", "x_sup"]
    )

    # The active factor space ignores supplementary cols, so dropping them up-front
    # and fitting must give the same eigenvalues + active row coords as passing them
    # as supplementary.
    famd_active = prince.FAMD(n_components=4, engine="scipy").fit(
        df.drop(columns=["q3_sup", "x_sup"])
    )
    np.testing.assert_allclose(famd_sup.eigenvalues_, famd_active.eigenvalues_, atol=1e-10)
    np.testing.assert_allclose(
        famd_sup.row_coordinates(df).abs().values,
        famd_active.row_coordinates(df.drop(columns=["q3_sup", "x_sup"])).abs().values,
        atol=1e-10,
    )

    # Supplementary cols (or their modalities) appear in column_coordinates_ but
    # don't affect eigenvalues.
    assert "x_sup" in famd_sup.column_coordinates_.index
    assert any(m.startswith("q3_sup_") for m in famd_sup.column_coordinates_.index)
    assert "x_sup" not in famd_full.cat_cols_

    # num_cols_/cat_cols_ keep the *full* picture; active subsets are derivable.
    assert "x_sup" in famd_sup.num_cols_
    assert "q3_sup" in famd_sup.cat_cols_
    active_num = list(famd_sup.num_scaler_.feature_names_in_)
    assert "x_sup" not in active_num
