"""Multiple Factor Analysis (MFA)"""
import collections
import itertools

import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder

from prince import mca
from prince import pca
from prince import plot
from prince import utils


class MFA(pca.PCA, collections.UserDict):
    def __init__(
        self,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        as_array=False,
    ):
        super().__init__(
            rescale_with_mean=False,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
            as_array=as_array,
        )
        collections.UserDict.__init__(self)

    def fit(self, X, y=None, groups=None):

        # Checks groups are provided
        self.groups_ = self._determine_groups(X, groups)

        # Check input
        if self.check_input:
            check_array(X, dtype=[str, np.number])

        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups_.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(
                    'Not all columns in "{}" group are of the same type'.format(name)
                )
            self.all_nums_[name] = all_num

        # Normalize column-wise
        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5

        # Run a factor analysis in each group
        for name, cols in sorted(self.groups_.items()):
            if self.all_nums_[name]:
                fa = pca.PCA(
                    rescale_with_mean=True,
                    rescale_with_std=True,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine,
                )
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine,
                )
            self[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        Z = pd.concat(
            (
                X[cols].copy() / self[g].eigenvalues_[0] ** 0.5
                for g, cols in self.groups_.items()
            ),
            axis="columns",
        )
        super().fit(Z)
        self.total_inertia_ = sum(self.eigenvalues_)

        return self

    def _determine_groups(self, X, provided_groups):
        if provided_groups is None:
            raise ValueError("Groups have to be specified")
        if isinstance(provided_groups, list):
            if not isinstance(X.columns, pd.MultiIndex):
                raise ValueError(
                    "Groups have to be provided as a dict when X is not a MultiIndex"
                )
            groups = {
                g: [
                    (g, c)
                    for c in X.columns.get_level_values(1)[
                        X.columns.get_level_values(0) == g
                    ]
                ]
                for g in provided_groups
            }
        else:
            groups = provided_groups
        return groups

    # def _prepare_input(self, X):

    #     # Make sure X is a DataFrame for convenience
    #     if not isinstance(X, pd.DataFrame):
    #         X = pd.DataFrame(X)

    #     # Copy data
    #     if self.copy:
    #         X = X.copy()

    #     # if self.normalize:
    #     #     # Scale continuous variables to unit variance
    #     #     num = X.select_dtypes(np.number).columns
    #     #     # If a column's cardinality is 1 then it's variance is 0 which can
    #     #     # can cause a division by 0
    #     #     normalize = lambda x: x / (np.sqrt((x**2).sum()) or 1)
    #     #     X.loc[:, num] = (X.loc[:, num] - X.loc[:, num].mean()).apply(
    #     #         normalize, axis="rows"
    #     #     )

    #     return X

    # def _build_X_global(self, X):
    #     X_partials = []

    #     for name, cols in sorted(self.groups.items()):
    #         X_partial = X.loc[:, cols]

    #         # Dummify if there are categorical variable
    #         if not self.all_nums_[name]:

    #             # From FactoMineR MFA code, needs checking
    #             try:
    #                 tmp = pd.DataFrame(self.enc.transform(X_partial))
    #             except AttributeError:
    #                 self.enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    #                 self.enc.fit(X_partial)
    #                 tmp = pd.DataFrame(self.enc.transform(X_partial))
    #             centre_tmp = tmp.mean() / len(tmp)
    #             tmp2 = tmp / len(tmp)
    #             poids_bary = tmp2.sum()
    #             poids_tmp = 1 - tmp2.sum()
    #             ponderation = poids_tmp**0.5 / (
    #                 self.partial_factor_analysis_[name].s_[0] * len(cols)
    #             )

    #             normalize = lambda x: x / (np.sqrt((x**2).sum()) or 1)
    #             tmp = (tmp - tmp.mean()).apply(normalize, axis="rows")

    #             X_partial = tmp
    #             X_partial *= ponderation**0.5

    #             X_partials.append(X_partial)

    #         else:

    #             X_partials.append(X_partial / self.partial_factor_analysis_[name].s_[0])

    #     X_global = pd.concat(X_partials, axis="columns")
    #     X_global.index = X.index
    #     return X_global

    # def transform(self, X):
    #     """Returns the row principal coordinates of a dataset."""
    #     return self.row_coordinates(X)

    # def _row_coordinates_from_global(self, X_global):
    #     """Returns the row principal coordinates."""
    #     return len(X_global) ** 0.5 * super().row_coordinates(X_global)

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s)

    def row_coordinates(self, X):
        """Returns the row principal coordinates."""

        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5
        Z = pd.concat(
            (
                X[cols].copy() / self[g].eigenvalues_[0] ** 0.5
                for g, cols in self.groups_.items()
            ),
            axis="columns",
        )
        U = self.svd_.U
        s = self.svd_.s
        M = np.full(len(X), 1 / len(X))
        return (Z @ Z.T) @ (M[:, np.newaxis] ** (-0.5) * U * s**-1)

    # def row_contributions(self, X):
    #     """Returns the row contributions towards each principal component."""
    #     self._check_is_fitted()

    #     # Check input
    #     if self.check_input:
    #         utils.check_array(X, dtype=[str, np.number])

    #     # Prepare input
    #     X = self._prepare_input(X)

    #     return super().row_contributions(self._build_X_global(X))

    # def partial_row_coordinates(self, X):
    #     """Returns the row coordinates for each group."""
    #     self._check_is_fitted()

    #     # Check input
    #     if self.check_input:
    #         utils.check_array(X, dtype=[str, np.number])

    #     # Prepare input
    #     X = self._prepare_input(X)

    #     # Define the projection matrix P
    #     P = len(X) ** 0.5 * self.U_ / self.s_

    #     # Get the projections for each group
    #     coords = {}
    #     for name, cols in sorted(self.groups.items()):
    #         X_partial = X.loc[:, cols]

    #         if not self.all_nums_[name]:
    #             X_partial = pd.DataFrame(self.enc.transform(X_partial))

    #         Z_partial = X_partial / self.partial_factor_analysis_[name].s_[0]
    #         coords[name] = len(self.groups) * (Z_partial @ Z_partial.T) @ P

    #     # Convert coords to a MultiIndex DataFrame
    #     coords = pd.DataFrame(
    #         {
    #             (name, i): group_coords.loc[:, i]
    #             for name, group_coords in coords.items()
    #             for i in range(group_coords.shape[1])
    #         }
    #     )

    #     return coords

    # def column_correlations(self, X):
    #     """Returns the column correlations."""
    #     self._check_is_fitted()

    #     X_global = self._build_X_global(X)
    #     row_pc = self._row_coordinates_from_global(X_global)

    #     return pd.DataFrame(
    #         {
    #             component: {
    #                 feature: row_pc[component].corr(X_global[feature])
    #                 for feature in X_global.columns
    #             }
    #             for component in row_pc.columns
    #         }
    #     ).sort_index()

    # def plot_partial_row_coordinates(
    #     self,
    #     X,
    #     ax=None,
    #     figsize=(6, 6),
    #     x_component=0,
    #     y_component=1,
    #     color_labels=None,
    #     **kwargs
    # ):
    #     """Plot the row principal coordinates."""
    #     self._check_is_fitted()

    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     # Add plotting style
    #     ax = plot.stylize_axis(ax)

    #     # Check input
    #     if self.check_input:
    #         utils.check_array(X, dtype=[str, np.number])

    #     # Prepare input
    #     X = self._prepare_input(X)

    #     # Retrieve partial coordinates
    #     coords = self.partial_row_coordinates(X)

    #     # Determine the color of each group if there are group labels
    #     if color_labels is not None:
    #         colors = {
    #             g: ax._get_lines.get_next_color()
    #             for g in sorted(list(set(color_labels)))
    #         }

    #     # Get the list of all possible markers
    #     marks = itertools.cycle(list(markers.MarkerStyle.markers.keys()))
    #     next(marks)  # The first marker looks pretty shit so we skip it

    #     # Plot points
    #     for name in self.groups:

    #         mark = next(marks)

    #         x = coords[name][x_component]
    #         y = coords[name][y_component]

    #         if color_labels is None:
    #             ax.scatter(x, y, marker=mark, label=name, **kwargs)
    #             continue

    #         for color_label, color in sorted(colors.items()):
    #             mask = np.array(color_labels) == color_label
    #             label = "{} - {}".format(name, color_label)
    #             ax.scatter(
    #                 x[mask], y[mask], marker=mark, color=color, label=label, **kwargs
    #             )

    #     # Legend
    #     ax.legend()

    #     # Text
    #     ax.set_title("Partial row principal coordinates")
    #     ei = self.explained_inertia_
    #     ax.set_xlabel(
    #         "Component {} ({:.2f}% inertia)".format(x_component, 100 * ei[x_component])
    #     )
    #     ax.set_ylabel(
    #         "Component {} ({:.2f}% inertia)".format(y_component, 100 * ei[y_component])
    #     )

    #     return ax
