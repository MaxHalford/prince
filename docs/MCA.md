# Multiple correspondence analysis (MCA)

## When to use it?

You have more than 2 variables and they are all categorical.

## Learning material

- [Computation of Multiple Correspondence Analysis, with code in R](https://core.ac.uk/download/pdf/6591520.pdf)

## User guide


Multiple correspondence analysis (MCA) is an extension of correspondence analysis (CA). It should be used when you have more than two categorical variables. The idea is simply to compute the one-hot encoded version of a dataset and apply CA on it. As an example we're going to use the [balloons dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/) taken from the [UCI datasets website](https://archive.ics.uci.edu/ml/datasets.html).

```python
>>> import pandas as pd

>>> X = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data')
>>> X.columns = ['Color', 'Size', 'Action', 'Age', 'Inflated']
>>> X.head()
    Color   Size   Action    Age Inflated
0  YELLOW  SMALL  STRETCH  ADULT        T
1  YELLOW  SMALL  STRETCH  CHILD        F
2  YELLOW  SMALL      DIP  ADULT        F
3  YELLOW  SMALL      DIP  CHILD        F
4  YELLOW  LARGE  STRETCH  ADULT        T

```

The `MCA` also implements the `fit` and `transform` methods.

```python
>>> import prince
>>> mca = prince.MCA(
...     n_components=2,
...     n_iter=3,
...     copy=True,
...     check_input=True,
...     engine='sklearn',
...     random_state=42
... )
>>> mca = mca.fit(X)

```

Like the `CA` class, the `MCA` class also has `plot_coordinates` method.

```python
>>> ax = mca.plot_coordinates(
...     X=X,
...     ax=None,
...     figsize=(6, 6),
...     show_row_points=True,
...     row_points_size=10,
...     show_row_labels=False,
...     row_groups=None,
...     show_column_points=True,
...     column_points_size=30,
...     show_column_labels=False,
...     legend_n_cols=1
... )
>>> ax.get_figure().savefig('images/mca_coordinates.svg')

```

<div align="center">
  <img src="images/mca_coordinates.svg" />
</div>

The optional parameter `row_groups` takes a list of labels for coloring the observations. This list must have the same lenght than the amount of observations. If no list of labels is passed, then all observations are grey.

```python
>>> groups = ['CAT_A']*10+['CAT_B']*9
>>> ax = mca.plot_coordinates(
...     X=X,
...     ax=None,
...     figsize=(6, 6),
...     show_row_points=True,
...     row_points_size=10,
...     show_row_labels=False,
...     row_groups=groups,
...     show_column_points=True,
...     column_points_size=30,
...     show_column_labels=False,
...     legend_n_cols=1
... )
>>> ax.get_figure().savefig('images/mca_coordinates_with_groups.svg')

```

<div align="center">
  <img src="images/mca_coordinates_with_groups.svg" />
</div>

The eigenvalues and inertia values are also accessible.

```python
>>> mca.eigenvalues_
[0.401656..., 0.211111...]

>>> mca.total_inertia_
1.0

>>> mca.explained_inertia_
[0.401656..., 0.211111...]

```
