# Correspondence analysis (CA)

## When to use it?

You have a contingency table.

## Learning material

- [Theory of Correspondence Analysis](http://statmath.wu.ac.at/courses/CAandRelMeth/caipA.pdf)
- [Correspondence analysis -- Hervé Abdi & Michael Béra](https://cedric.cnam.fr/fichiers/art_3066.pdf)

## User guide

You should be using correspondence analysis when you want to analyse a contingency table. In other words you want to analyse the dependencies between two categorical variables. The following example comes from section 17.2.3 of [this textbook](http://ce.aut.ac.ir/~shiry/lecture/Advanced%20Machine%20Learning/Manifold_Modern_Multivariate%20Statistical%20Techniques%20-%20Regres.pdf). It shows the number of occurrences between different hair and eye colors.

```python
>>> import pandas as pd

>>> pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
>>> X = pd.DataFrame(
...    data=[
...        [326, 38, 241, 110, 3],
...        [688, 116, 584, 188, 4],
...        [343, 84, 909, 412, 26],
...        [98, 48, 403, 681, 85]
...    ],
...    columns=pd.Series(['Fair', 'Red', 'Medium', 'Dark', 'Black']),
...    index=pd.Series(['Blue', 'Light', 'Medium', 'Dark'])
... )
>>> X
        Fair  Red  Medium  Dark  Black
Blue     326   38     241   110      3
Light    688  116     584   188      4
Medium   343   84     909   412     26
Dark      98   48     403   681     85

```

Unlike the `PCA` class, the `CA` only exposes scikit-learn's `fit` method.

```python
>>> import prince
>>> ca = prince.CA(
...     n_components=2,
...     n_iter=3,
...     copy=True,
...     check_input=True,
...     engine='sklearn',
...     random_state=42
... )
>>> X.columns.rename('Hair color', inplace=True)
>>> X.index.rename('Eye color', inplace=True)
>>> ca = ca.fit(X)

```

The parameters and methods overlap with those proposed by the `PCA` class.

```python
>>> ca.row_coordinates(X)
               0         1
Blue   -0.400300 -0.165411
Light  -0.440708 -0.088463
Medium  0.033614  0.245002
Dark    0.702739 -0.133914

>>> ca.column_coordinates(X)
               0         1
Fair   -0.543995 -0.173844
Red    -0.233261 -0.048279
Medium -0.042024  0.208304
Dark    0.588709 -0.103950
Black   1.094388 -0.286437

```

You can plot both sets of principal coordinates with the `plot_coordinates` method.

```python
>>> ax = ca.plot_coordinates(
...     X=X,
...     ax=None,
...     figsize=(6, 6),
...     x_component=0,
...     y_component=1,
...     show_row_labels=True,
...     show_col_labels=True
... )
>>> ax.get_figure().savefig('images/ca_coordinates.svg')

```

<div align="center">
  <img src="images/ca_coordinates.svg" />
</div>

Like for the `PCA` you can access the inertia contribution of each principal component as well as the eigenvalues and the total inertia.

```python
>>> ca.eigenvalues_
[0.199244..., 0.030086...]

>>> ca.total_inertia_
0.230191...

>>> ca.explained_inertia_
[0.865562..., 0.130703...]

```
