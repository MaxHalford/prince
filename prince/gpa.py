"""Generalized Procrustes Analysis (GPA)"""
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn import base
from sklearn import utils


class GPA(base.BaseEstimator, base.TransformerMixin):
    """Generalized Procrustes Analysis (GPA).

    Algorithm outline:
        (https://wikipedia.org/wiki/Generalized_Procrustes_analysis)
    1. Choose a reference shape.
    2. Apply Procrustes Analysis to superimpose all shapes to the reference
        shape.
    3. Compute the mean shape of the superimposed shapes.
    4. Repeat steps 2 and 3 until convergence.

    Parameters:
        max_iter (int): The maximum number of Procrustes analysis iterations.
        tol (float): The tolerance for the optimization; stops if the
            procrustes distance decreases by less or equal to ``tol`` between
            iterations.
        init ({'random', 'mean'}): Method for initializing reference shape.
            'random' : choose reference shape from shape list
            'mean' : initialize reference shape as mean of shape list
        copy (bool): Whether to copy data or perform the computations inplace.
            If False, data passed to fit are overwritten and running
            fit(X).transform(X) will not yield the expected results,
            use fit_transform(X) instead.
        check_input (bool): Whether to check the consistency of the inputs.
        random_state (int, RandomState instance or None): Determines random
            number generation for initialization when ``init=='random'``

    """

    def __init__(
        self,
        max_iter=10,
        tol=1e-4,
        init='random',
        copy=True,
        check_input=True,
        random_state=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        The algorithm naturally fits and transforms at the same time, so this
        simply calls ``.fit_transform``

        Parameters:
            X (array-like of shape (n_shapes, n_points, n_dim)): Matrix of
                shapes to match to each other.
            y: Ignored

        Returns:
            self (object): The instance itself
        """
        self.fit_transform(X)

        return self

    def transform(self, X):
        """Align X to the reference shape.

        Parameters:
            X (array-like of shape (n_shapes, n_points, n_dim)): Matrix of
                shapes to align to the refernce shape.

        Returns:
            X_new (array-like of shape (n_shapes, n_points, n_dim)): Matrix of
                aligned shapes
        """
        self._check_is_fitted()
        if self.check_input:
            self._check_input(X)

        X_new = np.empty(X.shape)
        for shape_idx in range(X.shape[0]):
            _, X_new[shape_idx], _ = procrustes(self.reference_shape, X[shape_idx])

        return X_new

    def fit_transform(self, X, y=None):
        """Fit the model with X and return the aligned shapes.

        Parameters:
            X (array-like of shape (n_shapes, n_points, n_dim)): Matrix of
                shapes to match to each other.
            y: Ignored

        Returns:
            X_new (array-like of shape (n_shapes, n_points, n_dim)): Matrix X
                of aligned shapes
        """

        # Check input
        if self.check_input:
            self._check_input(X)

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # scikit-learn SLEP010
        n_shapes, n_points, n_dim = X.shape
        self.n_features_in_ = n_dim

        # Pick reference shape
        if self.init == 'random':
            random_state = utils.check_random_state(self.random_state)
            ref_shape_idx = random_state.randint(X.shape[0])
            reference_shape = X[ref_shape_idx]
        elif self.init == 'mean':
            reference_shape = X.mean(axis=0)
        else:
            raise ValueError("init method must be one of ('random', 'mean')")

        for iter_idx in range(self.max_iter):
            # Align each shape to reference shape
            for shape_idx in range(X.shape[0]):
                _, X[shape_idx], _ = procrustes(reference_shape, X[shape_idx])

            # Compute diagnostics
            mean_shape = X.mean(axis=0)
            _, _, disparity = procrustes(reference_shape, mean_shape)
            procrustes_distance = np.sqrt(disparity)

            # Update reference shape
            reference_shape = mean_shape

            # Check for convergence
            if procrustes_distance <= self.tol:
                break

        # Store properties
        self._reference_shape = reference_shape

        # Return the aligned shapes
        return X

    def _check_input(self, X):
        utils.check_array(X, allow_nd=True)
        if X.ndim != 3:
            raise ValueError(
                'Expected 3-dimensional input of (n_shapes, n_points, n_dim)'
            )

    def _check_is_fitted(self):
        utils.validation.check_is_fitted(self, '_reference_shape')

    @property
    def reference_shape(self):
        """Returns the final reference shape."""
        self._check_is_fitted()
        return self._reference_shape
