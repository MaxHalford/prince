import unittest

import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn.utils import estimator_checks

import prince


class TestGPA(unittest.TestCase):

    # def setUp(self):
    def __init__(self):
        # Create a list of 2-D circles with different locations and rotations
        n_shapes = 4
        n_points = 12
        n_dims = 2

        shape_sizes = np.arange(1, n_shapes + 1)
        shape_angle_offsets = 10 * np.arange(n_shapes)
        shape_center_offsets = np.tile(np.arange(n_shapes), (n_dims, 1))

        base_angles = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
        # Size (n_shapes, n_points)
        angles = base_angles[np.newaxis, :] + shape_angle_offsets[:, np.newaxis]

        # Calculate along dimensions
        x = (
            np.cos(angles) * shape_sizes[:, np.newaxis]
            + shape_center_offsets[0][:, np.newaxis]
        )
        y = (
            np.sin(angles) * shape_sizes[:, np.newaxis]
            + shape_center_offsets[1][:, np.newaxis]
        )

        self.shapes = np.stack([x, y], axis=-1)

    def test_fit(self):
        gpa = prince.GPA()
        self.assertIsInstance(gpa.fit(self.shapes), prince.GPA)

    def test_transform(self):
        gpa = prince.GPA(copy=True)
        aligned_shapes = gpa.fit(self.shapes).transform(self.shapes)
        self.assertIsInstance(aligned_shapes, np.ndarray)
        self.assertEqual(self.shapes.shape, aligned_shapes.shape)

    def test_fit_transform(self):
        gpa = prince.GPA()
        aligned_shapes = gpa.fit_transform(self.shapes)
        self.assertIsInstance(aligned_shapes, np.ndarray)

    def test_fit_transform_single(self):
        """Aligning a single shape should return the same shape."""
        gpa = prince.GPA()
        shapes = self.shapes.shape[0:1]
        aligned_shapes = gpa.fit_transform(shapes)
        np.testing.assert_array_equal(shapes, aligned_shapes)

    def test_copy(self):
        shapes_copy = np.copy(self.shapes)

        gpa = prince.GPA(copy=True)
        gpa.fit(shapes_copy)
        np.testing.assert_array_equal(self.shapes, shapes_copy)

        gpa = prince.GPA(copy=False)
        gpa.fit(shapes_copy)
        self.assertRaises(
            AssertionError, np.testing.assert_array_equal, self.shapes, shapes_copy
        )

    def test_check_estimator(self):
        estimator_checks.check_estimator(prince.GPA(as_array=True))
