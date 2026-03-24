"""Principal Geodesic Analysis (PGA)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base

from prince import pca as _pca
from prince import utils
from prince.manifolds import SO3


class PGA(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, utils.EigenvaluesMixin):
    """Principal Geodesic Analysis (PGA).

    PGA generalizes PCA to Riemannian manifolds. Data is mapped to the tangent
    space at the Fréchet mean, PCA is performed there, and results can be mapped
    back to the manifold.

    Currently supports SO(3) (3D rotations) with quaternion input.

    Parameters
    ----------
    n_components
        The number of principal geodesic components to compute.
    manifold
        The manifold to use. Currently only "SO3" is supported.
    input_format
        The input format. Currently only "quaternion" is supported, expecting
        columns (qw, qx, qy, qz) in scalar-first convention.
    mean_max_iter
        Maximum iterations for Fréchet mean computation.
    mean_tol
        Convergence tolerance for Fréchet mean computation.
    n_iter
        The number of iterations used for computing the SVD.
    random_state
        Random state for reproducibility.
    engine
        The SVD engine to use. See prince.PCA for options.

    """

    def __init__(
        self,
        n_components=2,
        manifold="SO3",
        input_format="quaternion",
        mean_max_iter=50,
        mean_tol=1e-7,
        n_iter=3,
        random_state=None,
        engine="sklearn",
    ):
        self.n_components = n_components
        self.manifold = manifold
        self.input_format = input_format
        self.mean_max_iter = mean_max_iter
        self.mean_tol = mean_tol
        self.n_iter = n_iter
        self.random_state = random_state
        self.engine = engine

    def _get_manifold(self):
        if self.manifold == "SO3":
            return SO3()
        raise ValueError(f"Unknown manifold: {self.manifold!r}. Supported: 'SO3'.")

    def _parse_quaternions(self, X):
        """Extract quaternion array (n, 4) in scalar-first format from DataFrame."""
        if isinstance(X, pd.DataFrame):
            if {"qw", "qx", "qy", "qz"}.issubset(X.columns):
                return X[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
            if X.shape[1] == 4:
                return X.to_numpy(dtype=np.float64)
            raise ValueError(
                "Expected DataFrame with columns (qw, qx, qy, qz) or exactly 4 columns."
            )
        return np.asarray(X, dtype=np.float64)

    def _log_map_to_df(self, quats, index=None):
        """Log-map quaternions to tangent space and return as DataFrame."""
        tangent = self.manifold_.log(self.frechet_mean_, quats)
        return pd.DataFrame(tangent, columns=["rx", "ry", "rz"], index=index)

    def fit(self, X, y=None, sample_weight=None):
        """Fit PGA to data.

        Parameters
        ----------
        X
            Rotation data. A DataFrame with columns (qw, qx, qy, qz) or a
            (n, 4) array of unit quaternions in scalar-first format.
        y
            Ignored.
        sample_weight
            Optional sample weights.

        """
        quats = self._parse_quaternions(X)
        self.manifold_ = self._get_manifold()

        # Normalize quaternions
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Compute Fréchet mean
        self.frechet_mean_ = self.manifold_.frechet_mean(
            quats,
            weights=sample_weight,
            max_iter=self.mean_max_iter,
            tol=self.mean_tol,
        )

        # Log-map to tangent space
        index = X.index if isinstance(X, pd.DataFrame) else None
        tangent_df = self._log_map_to_df(quats, index=index)

        # Delegate to PCA
        self.pca_ = _pca.PCA(
            n_components=self.n_components,
            rescale_with_mean=True,
            rescale_with_std=False,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine,
        )
        self.pca_.fit(tangent_df, sample_weight=sample_weight)

        return self

    @utils.check_is_fitted
    def transform(self, X):
        """Project rotation data onto principal geodesic components.

        Parameters
        ----------
        X
            Rotation data (same format as fit).

        Returns
        -------
        pd.DataFrame with projected coordinates.

        """
        quats = self._parse_quaternions(X)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        index = X.index if isinstance(X, pd.DataFrame) else None
        tangent_df = self._log_map_to_df(quats, index=index)
        return self.pca_.transform(tangent_df)

    @utils.check_is_fitted
    def inverse_transform(self, X):
        """Map projected coordinates back to quaternions on the manifold.

        Parameters
        ----------
        X
            Projected coordinates from transform().

        Returns
        -------
        pd.DataFrame with columns (qw, qx, qy, qz).

        """
        tangent_df = self.pca_.inverse_transform(X)
        tangent = tangent_df.to_numpy() if isinstance(tangent_df, pd.DataFrame) else tangent_df
        quats = self.manifold_.exp(self.frechet_mean_, tangent)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(quats, columns=["qw", "qx", "qy", "qz"], index=index)

    @utils.check_is_fitted
    def row_coordinates(self, X):
        """Returns the row principal coordinates.

        Equivalent to transform().

        """
        return self.transform(X)

    @utils.check_is_fitted
    def plot(
        self,
        X,
        x_component=0,
        y_component=1,
        color_rows_by=None,
        show_row_markers=True,
        show_row_labels=False,
    ):
        """Plot the projected data in the tangent space.

        Parameters
        ----------
        X
            Rotation data (same format as fit).
        x_component
            Component for the x-axis.
        y_component
            Component for the y-axis.
        color_rows_by
            Column name or Series to color points by.
        show_row_markers
            Whether to show point markers.
        show_row_labels
            Whether to show point labels.

        """
        quats = self._parse_quaternions(X)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        index = X.index if isinstance(X, pd.DataFrame) else None
        tangent_df = self._log_map_to_df(quats, index=index)

        # Put the color column into the index so PCA.plot can find it
        # after reset_index()
        if color_rows_by is not None and isinstance(X, pd.DataFrame):
            if isinstance(color_rows_by, str) and color_rows_by in X.columns:
                tangent_df = tangent_df.set_index(
                    pd.Index(X[color_rows_by].values, name=color_rows_by)
                )

        return self.pca_.plot(
            tangent_df,
            x_component=x_component,
            y_component=y_component,
            color_rows_by=color_rows_by,
            show_row_markers=show_row_markers,
            show_column_markers=False,
            show_row_labels=show_row_labels,
            show_column_labels=False,
        )

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Eigenvalues from PCA in the tangent space."""
        return self.pca_.eigenvalues_

    @property
    @utils.check_is_fitted
    def total_inertia_(self):
        """Total inertia from PCA in the tangent space."""
        return self.pca_.total_inertia_

    @property
    @utils.check_is_fitted
    def column_coordinates_(self):
        """Column coordinates from PCA in the tangent space."""
        return self.pca_.column_coordinates_
