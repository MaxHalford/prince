# Multiple factor analysis (MFA)

## When to use it?

## Learning material

- [Multiple Factor Analysis](https://www.utdallas.edu/~herve/Abdi-MFA2007-pretty.pdf)

## User guide


Multiple factor analysis (MFA) is meant to be used when you have groups of variables. In practice it builds a PCA on each group -- or an MCA, depending on the types of the group's variables. It then constructs a global PCA on the results of the so-called partial PCAs -- or MCAs. The dataset used in the following examples come from [this paper](https://www.utdallas.edu/~herve/Abdi-MFA2007-pretty.pdf). In the dataset, three experts give their opinion on six different wines. Each opinion for each wine is recorded as a variable. We thus want to consider the separate opinions of each expert whilst also having a global overview of each wine. MFA is the perfect fit for this kind of situation.

First of all let's copy the data used in the paper.

```python
>>> import pandas as pd

>>> X = pd.DataFrame(
...     data=[
...         [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
...         [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
...         [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
...         [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
...         [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
...         [3, 4, 4, 3, 5, 4, 5, 1, 7, 5]
...     ],
...     columns=['E1 fruity', 'E1 woody', 'E1 coffee',
...              'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
...              'E3 fruity', 'E3 butter', 'E3 woody'],
...     index=['Wine {}'.format(i+1) for i in range(6)]
... )
>>> X['Oak type'] = [1, 2, 2, 2, 1, 1]

```

The groups are passed as a dictionary to the `MFA` class.

```python
>>> groups = {
...    'Expert #{}'.format(no+1): [c for c in X.columns if c.startswith('E{}'.format(no+1))]
...    for no in range(3)
... }
>>> import pprint
>>> pprint.pprint(groups)
{'Expert #1': ['E1 fruity', 'E1 woody', 'E1 coffee'],
 'Expert #2': ['E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody'],
 'Expert #3': ['E3 fruity', 'E3 butter', 'E3 woody']}

```

Now we can fit an `MFA`.

```python
>>> import prince
>>> mfa = prince.MFA(
...     groups=groups,
...     n_components=2,
...     n_iter=3,
...     copy=True,
...     check_input=True,
...     engine='sklearn',
...     random_state=42
... )
>>> mfa = mfa.fit(X)

```

The `MFA` inherits from the `PCA` class, which entails that you have access to all it's methods and properties. The `row_coordinates` method will return the global coordinates of each wine.

```python
>>> mfa.row_coordinates(X)
               0         1
Wine 1 -2.172155 -0.508596
Wine 2  0.557017 -0.197408
Wine 3  2.317663 -0.830259
Wine 4  1.832557  0.905046
Wine 5 -1.403787  0.054977
Wine 6 -1.131296  0.576241

```

Just like for the `PCA` you can plot the row coordinates with the `plot_row_coordinates` method.

```python
>>> ax = mfa.plot_row_coordinates(
...     X,
...     ax=None,
...     figsize=(6, 6),
...     x_component=0,
...     y_component=1,
...     labels=X.index,
...     color_labels=['Oak type {}'.format(t) for t in X['Oak type']],
...     ellipse_outline=False,
...     ellipse_fill=True,
...     show_points=True
... )
>>> ax.get_figure().savefig('images/mfa_row_coordinates.svg')

```

<div align="center">
  <img src="images/mfa_row_coordinates.svg" />
</div>

You can also obtain the row coordinates inside each group. The `partial_row_coordinates` method returns a `pandas.DataFrame` where the set of columns is a `pandas.MultiIndex`. The first level of indexing corresponds to each specified group whilst the nested level indicates the coordinates inside each group.

```python
>>> mfa.partial_row_coordinates(X)  # doctest: +NORMALIZE_WHITESPACE
  Expert #1           Expert #2           Expert #3
               0         1         0         1         0         1
Wine 1 -2.764432 -1.104812 -2.213928 -0.863519 -1.538106  0.442545
Wine 2  0.773034  0.298919  0.284247 -0.132135  0.613771 -0.759009
Wine 3  1.991398  0.805893  2.111508  0.499718  2.850084 -3.796390
Wine 4  1.981456  0.927187  2.393009  1.227146  1.123206  0.560803
Wine 5 -1.292834 -0.620661 -1.492114 -0.488088 -1.426414  1.273679
Wine 6 -0.688623 -0.306527 -1.082723 -0.243122 -1.622541  2.278372

```

Likewhise you can visualize the partial row coordinates with the `plot_partial_row_coordinates` method.

```python
>>> ax = mfa.plot_partial_row_coordinates(
...     X,
...     ax=None,
...     figsize=(6, 6),
...     x_component=0,
...     y_component=1,
...     color_labels=['Oak type {}'.format(t) for t in X['Oak type']]
... )
>>> ax.get_figure().savefig('images/mfa_partial_row_coordinates.svg')

```

<div align="center">
  <img src="images/mfa_partial_row_coordinates.svg" />
</div>

As usual you have access to inertia information.

```python
>>> mfa.eigenvalues_
array([0.47246678, 0.05947651])

>>> mfa.total_inertia_
0.558834...

>>> mfa.explained_inertia_
array([0.84545097, 0.10642965])

```

You can also access information concerning each partial factor analysis via the `partial_factor_analysis_` attribute.

```python
>>> for name, fa in sorted(mfa.partial_factor_analysis_.items()):
...     print('{} eigenvalues: {}'.format(name, fa.eigenvalues_))
Expert #1 eigenvalues: [0.47709918 0.01997272]
Expert #2 eigenvalues: [0.60851399 0.03235984]
Expert #3 eigenvalues: [0.41341481 0.07353257]

```

The `row_contributions` method will provide you with the inertia contribution of each row with respect to each component.

```python
>>> mfa.row_contributions(X)
               0         1
Wine 1  9.986433  4.349104
Wine 2  0.656699  0.655218
Wine 3 11.369187 11.589968
Wine 4  7.107942 13.771950
Wine 5  4.170915  0.050817
Wine 6  2.708824  5.582943

```

The `column_correlations` method will return the correlation between the original variables and the components.

```python
>>> mfa.column_correlations(X)
                     0         1
E1 coffee    -0.918449 -0.043444
E1 fruity     0.968449  0.192294
E1 woody     -0.984442 -0.120198
E2 red fruit  0.887263  0.357632
E2 roasted   -0.955795  0.026039
E2 vanillin  -0.950629 -0.177883
E2 woody     -0.974649  0.127239
E3 butter    -0.945767  0.221441
E3 fruity     0.594649 -0.820777
E3 woody     -0.992337  0.029747

```
