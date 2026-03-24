"""Riemannian manifold operations for Principal Geodesic Analysis."""

from __future__ import annotations

import abc

import numpy as np
from scipy.spatial.transform import Rotation


class Manifold(abc.ABC):
    """Abstract base class for Riemannian manifolds."""

    @abc.abstractmethod
    def log(self, base, points):
        """Logarithmic map: project manifold points to tangent space at base.

        Parameters
        ----------
        base
            The base point on the manifold.
        points
            Points on the manifold to project.

        Returns
        -------
        Tangent vectors at the base point.

        """

    @abc.abstractmethod
    def exp(self, base, tangent_vectors):
        """Exponential map: project tangent vectors back to the manifold.

        Parameters
        ----------
        base
            The base point on the manifold.
        tangent_vectors
            Tangent vectors at the base point.

        Returns
        -------
        Points on the manifold.

        """

    @abc.abstractmethod
    def frechet_mean(self, points, weights=None, max_iter=50, tol=1e-7):
        """Compute the Fréchet mean on the manifold.

        Parameters
        ----------
        points
            Points on the manifold.
        weights
            Optional weights for each point.
        max_iter
            Maximum number of iterations.
        tol
            Convergence tolerance.

        Returns
        -------
        The Fréchet mean.

        """

    @property
    @abc.abstractmethod
    def tangent_dim(self):
        """Dimension of the tangent space."""


class SO3(Manifold):
    """The SO(3) manifold of 3D rotations.

    Uses unit quaternions as the representation. Internally converts to/from
    scipy Rotation objects for Log/Exp map computations via rotation vectors.

    """

    @property
    def tangent_dim(self):
        return 3

    @staticmethod
    def _ensure_positive_hemisphere(base_quat, quats):
        """Ensure quaternions are in the same hemisphere as the base.

        Quaternions q and -q represent the same rotation, so we pick the sign
        that gives a non-negative dot product with the base quaternion.

        """
        dots = np.sum(base_quat * quats, axis=1)
        signs = np.where(dots >= 0, 1.0, -1.0)
        return quats * signs[:, np.newaxis]

    def log(self, base, points):
        """Log map: compute rotation vectors of base^{-1} * points.

        Parameters
        ----------
        base
            Base quaternion (4,) in scalar-first format (w, x, y, z).
        points
            Quaternions (n, 4) in scalar-first format.

        Returns
        -------
        np.ndarray of shape (n, 3) — tangent vectors (rotation vectors).

        """
        points = self._ensure_positive_hemisphere(base, points)
        base_rot = Rotation.from_quat(base[..., [1, 2, 3, 0]])  # scipy uses scalar-last
        point_rots = Rotation.from_quat(points[..., [1, 2, 3, 0]])
        relative = base_rot.inv() * point_rots
        return relative.as_rotvec()

    def exp(self, base, tangent_vectors):
        """Exp map: apply rotation vectors to base.

        Parameters
        ----------
        base
            Base quaternion (4,) in scalar-first format (w, x, y, z).
        tangent_vectors
            Tangent vectors (n, 3) — rotation vectors.

        Returns
        -------
        np.ndarray of shape (n, 4) — quaternions in scalar-first format.

        """
        base_rot = Rotation.from_quat(base[..., [1, 2, 3, 0]])
        delta_rots = Rotation.from_rotvec(tangent_vectors)
        result = base_rot * delta_rots
        q = result.as_quat()  # scalar-last (x, y, z, w)
        return q[:, [3, 0, 1, 2]]  # convert to scalar-first (w, x, y, z)

    def frechet_mean(self, points, weights=None, max_iter=50, tol=1e-7):
        """Compute the Fréchet mean via iterative Riemannian gradient descent.

        Parameters
        ----------
        points
            Quaternions (n, 4) in scalar-first format (w, x, y, z).
        weights
            Optional weights (n,). Defaults to uniform.
        max_iter
            Maximum iterations.
        tol
            Convergence tolerance on the tangent vector norm.

        Returns
        -------
        np.ndarray of shape (4,) — the Fréchet mean quaternion.

        """
        n = len(points)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()

        # Initialize with the first point
        mu = points[0].copy()

        for _ in range(max_iter):
            tangent_vecs = self.log(mu, points)
            avg_tangent = weights @ tangent_vecs
            if np.linalg.norm(avg_tangent) < tol:
                break
            # Step along the mean tangent direction
            mu_rot = Rotation.from_quat(mu[[1, 2, 3, 0]])
            step_rot = Rotation.from_rotvec(avg_tangent)
            mu_rot = mu_rot * step_rot
            q = mu_rot.as_quat()  # scalar-last
            mu = np.array([q[3], q[0], q[1], q[2]])

        return mu
