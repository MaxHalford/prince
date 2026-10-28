from __future__ import annotations

import inspect
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import prince

ESTIMATOR_TYPES = (
    prince.CA,
    prince.MCA,
    prince.PCA,
    prince.FAMD,
    prince.MFA,
    prince.GPA,
    prince.PGA,
)


def _boolean_configurations(estimator_type):
    parameters = [
        parameter.name
        for parameter in inspect.signature(estimator_type).parameters.values()
        if type(parameter.default) is bool
    ]
    for values in itertools.product([False, True], repeat=len(parameters)):
        yield dict(zip(parameters, values))


BOOLEAN_CONFIGURATION_CASES = [
    pytest.param(
        estimator_type,
        configuration,
        id="-".join(
            [estimator_type.__name__] + [f"{name}={value}" for name, value in configuration.items()]
        ),
    )
    for estimator_type in ESTIMATOR_TYPES
    for configuration in _boolean_configurations(estimator_type)
]


def _numeric_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.uniform(1, 10, size=(12, 4)), columns=list("abcd"))


def _categorical_frame():
    return pd.DataFrame(
        {
            "color": np.resize(["red", "blue", "green"], 12),
            "shape": np.resize(["round", "square"], 12),
        }
    )


def _mixed_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "length": rng.normal(size=12),
            "width": rng.normal(size=12),
            "color": np.resize(["red", "blue", "green"], 12),
            "shape": np.resize(["round", "square"], 12),
        }
    )


def _shapes():
    angles = np.linspace(0, 2 * np.pi, num=8, endpoint=False)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.stack([base, base @ Rotation.from_euler("z", 20, degrees=True).as_matrix()[:2, :2]])


def _quaternions():
    rng = np.random.default_rng(42)
    quaternions = Rotation.from_rotvec(rng.normal(scale=0.1, size=(12, 3))).as_quat()
    return pd.DataFrame(
        {
            "qw": quaternions[:, 3],
            "qx": quaternions[:, 0],
            "qy": quaternions[:, 1],
            "qz": quaternions[:, 2],
        }
    )


def _fit_and_project(estimator_type, configuration):
    if estimator_type is prince.CA:
        X = _numeric_frame()
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(X)
        return estimator.row_coordinates(X)
    if estimator_type is prince.MCA:
        X = _categorical_frame()
        if not configuration["one_hot"]:
            X = pd.get_dummies(X, dtype=float)
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(X)
        return estimator.transform(X)
    if estimator_type is prince.PCA:
        X = _numeric_frame()
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(X)
        return estimator.transform(X)
    if estimator_type is prince.FAMD:
        X = _mixed_frame()
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(X)
        return estimator.transform(X)
    if estimator_type is prince.MFA:
        X = _numeric_frame()
        groups = {"size": ["a", "b"], "position": ["c", "d"]}
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(
            X, groups=groups
        )
        return estimator.transform(X)
    if estimator_type is prince.GPA:
        X = _shapes()
        estimator = estimator_type(init="mean", max_iter=3, **configuration).fit(X)
        return estimator.transform(X)
    if estimator_type is prince.PGA:
        X = _quaternions()
        estimator = estimator_type(n_components=2, engine="scipy", **configuration).fit(X)
        return estimator.transform(X)
    raise AssertionError(f"No smoke-test data configured for {estimator_type.__name__}")


@pytest.mark.parametrize("estimator_type,configuration", BOOLEAN_CONFIGURATION_CASES)
def test_all_boolean_parameter_combinations_smoke(estimator_type, configuration):
    result = _fit_and_project(estimator_type, configuration)

    assert np.isfinite(np.asarray(result)).all()
