# Principal component analysis (PCA)

## When to use it?

All your variables are numeric.

## Learning material

- [Principal component analysis -- Hervé Abdi & Lynne J. Williams](https://personal.utdallas.edu/~herve/abdi-awPCA2010.pdf)
- [A Tutorial on Principal Component Analysis](https://arxiv.org/pdf/1404.1100.pdf)

## User guide

If you're using PCA it is assumed you have a dataframe consisting of numerical continuous variables. In this example we're going to be using the [Iris flower dataset](https://www.wikiwand.com/en/Iris_flower_data_set).

```python
>>> import pandas as pd
>>> import prince
>>> from sklearn import datasets

>>> X, y = datasets.load_iris(return_X_y=True)
>>> X = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
>>> y = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
>>> X.head()
   Sepal length  Sepal width  Petal length  Petal width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
2           4.7          3.2           1.3          0.2
3           4.6          3.1           1.5          0.2
4           5.0          3.6           1.4          0.2

```

The `PCA` class implements scikit-learn's `fit`/`transform` API. It's parameters have to passed at initialisation before calling the `fit` method.

```python
>>> pca = prince.PCA(
...     n_components=2,
...     n_iter=3,
...     rescale_with_mean=True,
...     rescale_with_std=True,
...     copy=True,
...     check_input=True,
...     engine='sklearn',
...     random_state=42
... )
>>> pca = pca.fit(X)

```

The available parameters are:

- `n_components`: the number of components that are computed. You only need two if your intention is to make a chart.
- `n_iter`: the number of iterations used for computing the SVD
- `rescale_with_mean`: whether to substract each column's mean
- `rescale_with_std`: whether to divide each column by it's standard deviation
- `copy`: if `False` then the computations will be done inplace which can have possible side-effects on the input data
- `engine`: what SVD engine to use (should be one of `['fbpca', 'sklearn']`)
- `random_state`: controls the randomness of the SVD results.

Once the `PCA` has been fitted, it can be used to extract the row principal coordinates as so:

```python
>>> pca.transform(X).head()  # same as pca.row_coordinates(X).head()
          0         1
0 -2.264703  0.480027
1 -2.080961 -0.674134
2 -2.364229 -0.341908
3 -2.299384 -0.597395
4 -2.389842  0.646835

```

Each column stands for a principal component whilst each row stands a row in the original dataset. You can display these projections with the `plot_row_coordinates` method:

```python
>>> ax = pca.plot_row_coordinates(
...     X,
...     ax=None,
...     figsize=(6, 6),
...     x_component=0,
...     y_component=1,
...     labels=None,
...     color_labels=y,
...     ellipse_outline=False,
...     ellipse_fill=True,
...     show_points=True
... )
>>> ax.get_figure().savefig('images/pca_row_coordinates.svg')

```

<div align="center">
  <img src="images/pca_row_coordinates.svg" />
</div>

Each principal component explains part of the underlying of the distribution. You can see by how much by using the accessing the `explained_inertia_` property:

```python
>>> pca.explained_inertia_
array([0.72962445, 0.22850762])

```

The explained inertia represents the percentage of the inertia each principal component contributes. It sums up to 1 if the `n_components` property is equal to the number of columns in the original dataset. you The explained inertia is obtained by dividing the eigenvalues obtained with the SVD by the total inertia, both of which are also accessible.

```python
>>> pca.eigenvalues_
array([2.91849782, 0.91403047])

>>> pca.total_inertia_
4.000000...

>>> pca.explained_inertia_
array([0.72962445, 0.22850762])

```

You can also obtain the correlations between the original variables and the principal components.

```python
>>> pca.column_correlations(X)
                     0         1
Petal length  0.991555  0.023415
Petal width   0.964979  0.064000
Sepal length  0.890169  0.360830
Sepal width  -0.460143  0.882716

```

You may also want to know how much each observation contributes to each principal component. This can be done with the `row_contributions` method.

```python
>>> pca.row_contributions(X).head()
          0         1
0  1.757369  0.252098
1  1.483777  0.497200
2  1.915225  0.127896
3  1.811606  0.390447
4  1.956947  0.457748

```

You can also transform row projections back into their original space by using the `inverse_transform` method.

```python
>>> pca.inverse_transform(pca.transform(X)).head()
          0         1         2         3
0  5.018949  3.514854  1.466013  0.251922
1  4.738463  3.030433  1.603913  0.272074
2  4.720130  3.196830  1.328961  0.167414
3  4.668436  3.086770  1.384170  0.182247
4  5.017093  3.596402  1.345411  0.206706

```
