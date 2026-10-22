from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import altair as alt
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from sklearn.utils import validation

P = ParamSpec("P")
R = TypeVar("R")


def check_is_fitted(method: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        validation.check_is_fitted(self)
        return method(self, *method_args, **method_kwargs)

    return _impl


def check_is_dataframe_input(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        X = args[1] if len(args) > 1 else kwargs["X"]
        frame = as_pandas_dataframe(X)
        if frame is None:
            raise ValueError(
                f"The X argument must be a supported DataFrame, but got {type(X).__name__}"
            )
        if len(args) > 1:
            args = (*args[:1], frame, *args[2:])
        else:
            kwargs["X"] = frame
        return func(*args, **kwargs)

    return wrapper


def as_narwhals_dataframe(X):
    """Return an eager Narwhals DataFrame for a supported native DataFrame."""
    frame = nw.from_native(X, eager_only=True, pass_through=True)
    return frame if isinstance(frame, nw.DataFrame) else None


def as_pandas_dataframe(X):
    """Convert a supported eager DataFrame to pandas without requiring PyArrow."""
    if isinstance(X, pd.DataFrame):
        return X
    frame = as_narwhals_dataframe(X)
    if frame is None:
        return None
    return pd.DataFrame({column: frame.get_column(column).to_numpy() for column in frame.columns})


def make_labels_and_names(X):
    if isinstance(X, pd.DataFrame):
        row_label = X.index.name if X.index.name else "Rows"
        row_names = X.index.tolist()
        col_label = X.columns.name if X.columns.name else "Columns"
        col_names = X.columns.tolist()
    else:
        row_label = "Rows"
        row_names = list(range(X.shape[0]))
        col_label = "Columns"
        col_names = list(range(X.shape[1]))

    return row_label, row_names, col_label, col_names


class EigenvaluesMixin(ABC):
    @property
    @abstractmethod
    def eigenvalues_(self) -> np.ndarray:
        """Return the eigenvalues supplied by the concrete estimator."""

    @property
    def total_inertia_(self) -> float:
        """Return the total inertia stored by the concrete estimator."""
        try:
            return self._total_inertia
        except AttributeError as error:
            raise AttributeError("total_inertia_ is not available before fitting") from error

    @total_inertia_.setter
    def total_inertia_(self, value: float) -> None:
        self._total_inertia = value

    @property
    @check_is_fitted
    def percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return 100 * self.eigenvalues_ / self.total_inertia_

    @property
    @check_is_fitted
    def cumulative_percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return np.cumsum(self.percentage_of_variance_)

    @property
    @check_is_fitted
    def _eigenvalues_summary(self):
        """Return a summary of the eigenvalues and their importance."""
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                r"% of variance": self.percentage_of_variance_,
                r"% of variance (cumulative)": self.cumulative_percentage_of_variance_,
            },
            index=pd.RangeIndex(0, len(self.eigenvalues_), name="component"),
        )

    @property
    def eigenvalues_summary(self):
        """Return a summary of the eigenvalues and their importance."""
        summary = self._eigenvalues_summary
        summary["% of variance"] /= 100
        summary["% of variance (cumulative)"] /= 100
        summary["eigenvalue"] = summary["eigenvalue"].map("{:,.3f}".format)
        summary["% of variance"] = summary["% of variance"].map("{:.2%}".format)
        summary["% of variance (cumulative)"] = summary["% of variance (cumulative)"].map(
            "{:.2%}".format
        )
        summary.index.name = "component"
        return summary

    def scree_plot(self):
        """Scree plot.

        References
        ----------
        https://en.wikipedia.org/wiki/Scree_plot

        """
        eig = self._eigenvalues_summary.reset_index().astype({"component": str})
        return (
            alt.Chart(eig)
            .mark_bar(size=10)
            .encode(
                x=alt.X("component", sort=eig["component"]),
                y="eigenvalue",
                tooltip=eig.columns.tolist(),
            )
        )
