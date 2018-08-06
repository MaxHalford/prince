"""Multiple Factor Analysis (MFA)"""
import itertools

from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils

from . import mca
from . import one_hot
from . import pca
from . import plot


class MFA(pca.PCA):

    def __init__(self, groups=None, normalize=True, n_components=2, n_iter=10,
                 copy=True, random_state=None, engine='auto'):
        super().__init__(
            rescale_with_mean=False,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            random_state=random_state,
            engine=engine
        )
        self.groups = groups
        self.normalize = normalize

    def fit(self, X, y=None):

        # Checks groups are provided
        if self.groups is None:
            raise ValueError('Groups have to be specified')

        # Check input
        utils.check_array(X, dtype=[str, np.number])

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
            self.all_nums_[name] = all_num

        # Scale continuous variables to unit variance
        if self.normalize:
            num = list(itertools.chain(*[
                cols for name, cols in self.groups.items()
                if self.all_nums_[name]
            ]))
            normalize = lambda x: x / np.sqrt((x ** 2).sum())
            X.loc[:, num] = (X.loc[:, num] - X.loc[:, num].mean()).apply(normalize, axis='rows')

        # Run a factor analysis in each group
        self.partial_factor_analysis_ = {}
        for name, cols in sorted(self.groups.items()):
            if self.all_nums_[name]:
                fa = pca.PCA(
                    rescale_with_mean=False,
                    rescale_with_std=False,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        super().fit(self._build_X_global(X))

        return self

    def _build_X_global(self, X):

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_partials = []
        self.cat_one_hots_ = {}

        for name, cols in sorted(self.groups.items()):
            X_partial = X.loc[:, cols]

            if not self.all_nums_[name]:
                oh = one_hot.OneHotEncoder().fit(X)
                X_partial = oh.fit_transform(X_partial)
                self.cat_one_hots_[name] = oh

            if not self.all_nums_[name]:
                X_partials.append(X_partial / self.partial_factor_analysis_[name].s_[0])

        return pd.concat(X_partials, axis='columns')

    def transform(self, X):
        """Returns the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X)
        return self.row_coordinates(X)

    def row_coordinates(self, X):
        """Returns the row principal coordinates."""
        utils.validation.check_is_fitted(self, 's_')
        n = X.shape[0]
        return n ** 0.5 * super().row_coordinates(self._build_X_global(X))

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return super().row_contributions(self._build_X_global(X))

    def partial_row_coordinates(self, X):
        """Returns the row coordinates for each group."""
        utils.validation.check_is_fitted(self, 's_')

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Define the projection matrix P
        n = X.shape[0]
        P = n ** 0.5 * self.U_ / self.s_

        # Get the projections for each group
        coords = {}
        for name, cols in sorted(self.groups.items()):
            X_partial = X.loc[:, cols]

            if not self.all_nums_[name]:
                X_partial = self.cat_one_hots_[name].transform(X_partial)

            Z_partial = X_partial / self.partial_factor_analysis_[name].s_[0]
            coords[name] = len(self.groups) * (Z_partial @ Z_partial.T) @ P

        # Convert coords to a MultiIndex DataFrame
        coords = pd.DataFrame({
            (name, i): group_coords.loc[:, i]
            for name, group_coords in coords.items()
            for i in range(group_coords.shape[1])
        })

        return coords

    def plot_partial_row_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                     color_labels=None, **kwargs):
        """Plot the row principal coordinates."""
        utils.validation.check_is_fitted(self, 's_')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Retrieve partial coordinates
        coords = self.partial_row_coordinates(X)

        # Determine the color of each group if there are group labels
        if color_labels is not None:
            colors = {g: ax._get_lines.get_next_color() for g in sorted(list(set(color_labels)))}

        # Get the list of all possible markers
        marks = itertools.cycle(list(markers.MarkerStyle.markers.keys()))
        next(marks)  # The first marker looks pretty shit so we skip it

        # Plot points
        for name in self.groups:

            mark = next(marks)

            x = coords[name][x_component]
            y = coords[name][y_component]

            if color_labels is None:
                ax.scatter(x, y, marker=mark, label=name, **kwargs)
                continue

            for color_label, color in sorted(colors.items()):
                mask = np.array(color_labels) == color_label
                label = '{} - {}'.format(name, color_label)
                ax.scatter(x[mask], y[mask], marker=mark, color=color, label=label, **kwargs)

        # Legend
        ax.legend()

        # Text
        ax.set_title('Partial row principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
