import unittest

import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn.utils import estimator_checks

import prince


class TestGPA(unittest.TestCase):
    def setUp(self):
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

    def test_fit_random(self):
        gpa = prince.GPA(init='random')
        self.assertIsInstance(gpa.fit(self.shapes), prince.GPA)

    def test_fit_mean(self):
        gpa = prince.GPA(init='mean')
        self.assertIsInstance(gpa.fit(self.shapes), prince.GPA)

    def test_fit_bad_init(self):
        gpa = prince.GPA(init='bad init type')

        with self.assertRaises(ValueError):
            gpa.fit(self.shapes)

    def test_fit_bad_input_size(self):
        gpa = prince.GPA()

        with self.assertRaises(ValueError):
            gpa.fit(self.shapes[0])

    def test_transform(self):
        gpa = prince.GPA(copy=True)
        aligned_shapes = gpa.fit(self.shapes).transform(self.shapes)
        self.assertIsInstance(aligned_shapes, np.ndarray)
        self.assertEqual(self.shapes.shape, aligned_shapes.shape)

    def test_fit_transform_equal(self):
        """In our specific case of all-same-shape circles, the shapes should
        align perfectly."""
        gpa = prince.GPA()
        aligned_shapes = gpa.fit_transform(self.shapes)
        self.assertIsInstance(aligned_shapes, np.ndarray)
        np.testing.assert_array_almost_equal(aligned_shapes[:-1], aligned_shapes[1:])

    def test_fit_transform_single(self):
        """Aligning a single shape should return the same shape, just normalized."""
        gpa = prince.GPA()
        shapes = self.shapes[0:1]
        aligned_shapes = gpa.fit_transform(shapes)
        np.testing.assert_array_almost_equal(
            shapes / np.linalg.norm(shapes), aligned_shapes
        )

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
