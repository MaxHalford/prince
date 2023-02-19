# Factor analysis of mixed data (FAMD)

## When to use it?

## Learning material

## User guide

A description is on it's way. This section is empty because I have to refactor the documentation a bit.

```python
>>> import pandas as pd

>>> X = pd.DataFrame(
...     data=[
...         ['A', 'A', 'A', 2, 5, 7, 6, 3, 6, 7],
...         ['A', 'A', 'A', 4, 4, 4, 2, 4, 4, 3],
...         ['B', 'A', 'B', 5, 2, 1, 1, 7, 1, 1],
...         ['B', 'A', 'B', 7, 2, 1, 2, 2, 2, 2],
...         ['B', 'B', 'B', 3, 5, 6, 5, 2, 6, 6],
...         ['B', 'B', 'A', 3, 5, 4, 5, 1, 7, 5]
...     ],
...     columns=['E1 fruity', 'E1 woody', 'E1 coffee',
...              'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
...              'E3 fruity', 'E3 butter', 'E3 woody'],
...     index=['Wine {}'.format(i+1) for i in range(6)]
... )
>>> X['Oak type'] = [1, 2, 2, 2, 1, 1]

```

Now we can fit an `FAMD`.

```python
>>> import prince
>>> famd = prince.FAMD(
...     n_components=2,
...     n_iter=3,
...     copy=True,
...     check_input=True,
...     engine='sklearn',
...     random_state=42
... )
>>> famd = famd.fit(X.drop('Oak type', axis='columns'))

```

The `FAMD` inherits from the `MFA` class, which entails that you have access to all it's methods and properties. The `row_coordinates` method will return the global coordinates of each wine.

```python
>>> famd.row_coordinates(X)
               0         1
Wine 1 -1.488689 -1.002711
Wine 2 -0.449783 -1.354847
Wine 3  1.774255 -0.258528
Wine 4  1.565402  0.016484
Wine 5 -0.349655  1.516425
Wine 6 -1.051531  1.083178

```

Just like for the `MFA` you can plot the row coordinates with the `plot_row_coordinates` method.

```python
>>> ax = famd.plot_row_coordinates(
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
>>> ax.get_figure().savefig('images/famd_row_coordinates.svg')

```

<div align="center">
  <img src="images/famd_row_coordinates.svg" />
</div>
