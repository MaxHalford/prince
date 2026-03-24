from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
from sklearn.utils.validation import check_is_fitted

import prince


@pytest.fixture
def rotations_df():
    return prince.datasets.load_rotations()


@pytest.fixture
def small_quats():
    """Small synthetic quaternion dataset with known structure."""
    rng = np.random.default_rng(0)
    # Rotations clustered around identity with most variance along z-axis
    n = 100
    rotvecs = np.column_stack(
        [
            rng.normal(0, 0.02, n),  # small x variance
            rng.normal(0, 0.05, n),  # medium y variance
            rng.normal(0, 0.15, n),  # large z variance
        ]
    )
    rots = Rotation.from_rotvec(rotvecs)
    q = rots.as_quat()  # scalar-last
    df = pd.DataFrame(
        {"qw": q[:, 3], "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2]},
    )
    return df


class TestPGAFit:
    def test_fit_returns_self(self, rotations_df):
        pga = prince.PGA(n_components=2)
        result = pga.fit(rotations_df)
        assert result is pga

    def test_is_fitted(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        check_is_fitted(pga)

    def test_frechet_mean_is_unit_quaternion(self, rotations_df):
        pga = prince.PGA(n_components=3).fit(rotations_df)
        norm = np.linalg.norm(pga.frechet_mean_)
        assert abs(norm - 1.0) < 1e-10

    def test_eigenvalues_descending(self, rotations_df):
        pga = prince.PGA(n_components=3).fit(rotations_df)
        eigs = pga.eigenvalues_
        assert all(eigs[i] >= eigs[i + 1] for i in range(len(eigs) - 1))

    def test_variance_percentages_sum(self, rotations_df):
        pga = prince.PGA(n_components=3).fit(rotations_df)
        total = pga.cumulative_percentage_of_variance_[-1]
        assert abs(total - 100.0) < 0.01


class TestPGATransform:
    def test_transform_shape(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        coords = pga.transform(rotations_df)
        assert coords.shape == (len(rotations_df), 2)

    def test_row_coordinates_equals_transform(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        t = pga.transform(rotations_df)
        rc = pga.row_coordinates(rotations_df)
        pd.testing.assert_frame_equal(t, rc)


class TestPGARoundTrip:
    def test_inverse_transform_roundtrip(self, small_quats):
        """With all 3 components, inverse_transform should nearly reconstruct input."""
        pga = prince.PGA(n_components=3).fit(small_quats)
        coords = pga.transform(small_quats)
        reconstructed = pga.inverse_transform(coords)

        # Compare as rotations (geodesic distance)
        original_quats = small_quats[["qw", "qx", "qy", "qz"]].to_numpy()
        recon_quats = reconstructed[["qw", "qx", "qy", "qz"]].to_numpy()

        for i in range(len(original_quats)):
            r1 = Rotation.from_quat(original_quats[i, [1, 2, 3, 0]])
            r2 = Rotation.from_quat(recon_quats[i, [1, 2, 3, 0]])
            angle = (r1.inv() * r2).magnitude()
            assert angle < 1e-6, f"Row {i}: angle={angle}"


class TestPGAProperties:
    def test_eigenvalues_summary(self, rotations_df):
        pga = prince.PGA(n_components=3).fit(rotations_df)
        summary = pga.eigenvalues_summary
        assert "eigenvalue" in summary.columns
        assert "% of variance" in summary.columns
        assert len(summary) == 3

    def test_scree_plot(self, rotations_df):
        pga = prince.PGA(n_components=3).fit(rotations_df)
        chart = pga.scree_plot()
        assert chart is not None

    def test_column_coordinates(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        cc = pga.column_coordinates_
        assert cc.shape == (3, 2)  # 3 tangent dims, 2 components


class TestPGAKnownStructure:
    def test_dominant_component_axis(self, small_quats):
        """The first component should align with the z-axis (largest variance)."""
        pga = prince.PGA(n_components=3).fit(small_quats)
        # The first column coordinate should have largest loading on rz
        cc = pga.column_coordinates_
        first_component = cc.iloc[:, 0].abs()
        assert first_component.idxmax() == "rz"


class TestPGAPlot:
    def test_plot_runs(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        chart = pga.plot(rotations_df)
        assert chart is not None

    def test_plot_with_color(self, rotations_df):
        pga = prince.PGA(n_components=2).fit(rotations_df)
        chart = pga.plot(rotations_df, color_rows_by="surface")
        assert chart is not None


class TestFrechetMean:
    def test_identity_cluster(self):
        """Fréchet mean of rotations near identity should be near identity."""
        rng = np.random.default_rng(42)
        n = 200
        rotvecs = rng.normal(0, 0.01, size=(n, 3))
        rots = Rotation.from_rotvec(rotvecs)
        q = rots.as_quat()  # scalar-last
        quats = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])

        from prince.manifolds import SO3

        so3 = SO3()
        mean = so3.frechet_mean(quats)
        # Mean should be close to identity
        angle = Rotation.from_quat(mean[[1, 2, 3, 0]]).magnitude()
        assert angle < 0.05


class TestGeomstatsComparison:
    """Compare PGA results against geomstats TangentPCA as a reference."""

    def test_explained_variance_ratio(self, small_quats):
        import geomstats.geometry.special_orthogonal as gso
        from geomstats.learning.pca import TangentPCA

        so3 = gso.SpecialOrthogonal(n=3, point_type="vector", equip=False)

        # Convert quaternions to rotation vectors for geomstats
        quats = small_quats[["qw", "qx", "qy", "qz"]].to_numpy()
        rots = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
        rotvecs = rots.as_rotvec()

        tpca = TangentPCA(so3, n_components=3)
        tpca.fit(rotvecs)
        gs_ratios = tpca.explained_variance_ratio_

        pga = prince.PGA(n_components=3).fit(small_quats)
        prince_ratios = pga.percentage_of_variance_ / 100.0

        # Ratios should be close (both do PCA in tangent space)
        np.testing.assert_allclose(
            np.sort(prince_ratios)[::-1],
            np.sort(gs_ratios)[::-1],
            atol=0.05,
        )

    def test_transformed_coordinates(self, small_quats):
        import geomstats.geometry.special_orthogonal as gso
        from geomstats.learning.pca import TangentPCA

        so3 = gso.SpecialOrthogonal(n=3, point_type="vector", equip=False)

        quats = small_quats[["qw", "qx", "qy", "qz"]].to_numpy()
        rots = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
        rotvecs = rots.as_rotvec()

        tpca = TangentPCA(so3, n_components=3)
        gs_coords = tpca.fit_transform(rotvecs)

        pga = prince.PGA(n_components=3).fit(small_quats)
        prince_coords = pga.transform(small_quats).to_numpy()

        # Compare absolute values (sign ambiguity in SVD)
        # Sort columns by variance to align components
        gs_var = np.var(gs_coords, axis=0)
        prince_var = np.var(prince_coords, axis=0)
        gs_order = np.argsort(gs_var)[::-1]
        prince_order = np.argsort(prince_var)[::-1]

        for gi, pi in zip(gs_order, prince_order):
            corr = np.abs(np.corrcoef(gs_coords[:, gi], prince_coords[:, pi])[0, 1])
            assert corr > 0.95, f"Component correlation too low: {corr}"
