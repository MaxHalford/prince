# Generalized procrustes analysis (GPA)

## When to use it?

## Learning material

- [Theory of Correspondence Analysis](http://statmath.wu.ac.at/courses/CAandRelMeth/caipA.pdf)
- [Correspondence analysis -- Hervé Abdi & Michael Béra](https://cedric.cnam.fr/fichiers/art_3066.pdf)

## User guide

```py

Generalized procrustes analysis (GPA) is a shape analysis tool that aligns and scales a set of shapes to a common reference. Here, the term "shape" means an *ordered* sequence of points. GPA iteratively 1) aligns each shape with a reference shape (usually the mean shape), 2) then updates the reference shape, 3) repeating until converged.

Note that the final rotation of the aligned shapes may vary between runs, based on the initialization.

Here is an example aligning a few right triangles:

```python
import pandas as pd

points = pd.DataFrame(
    data=[
        [0, 0, 0, 0],
        [0, 2, 0, 1],
        [1, 0, 0, 2],
        [3, 2, 1, 0],
        [1, 2, 1, 1],
        [3, 3, 1, 2],
        [0, 0, 2, 0],
        [0, 4, 2, 1],
        [2, 0, 2, 2],
    ],
    columns=['x', 'y', 'shape', 'point']
)
```

```py
alt.Chart(points).mark_line(opacity=0.5).encode(
    x='x',
    y='y',
    detail='shape',
    color='shape:N'
)
```

<div align="center">
  <img src="images/gpa_input_triangles.svg" />
</div>

We need to convert the dataframe to a 3-D numpy array of size (shapes, points, dims).
There are many ways to do this. Here, we use `xarray` as a helper package.

```python
ds = df.set_index(['shape', 'point']).to_xarray()
da = ds.to_stacked_array('xy', ['shape', 'point'])
shapes = da.values
```

Now, we can align the shapes.

```python
import prince
gpa = prince.GPA()
aligned_shapes = gpa.fit_transform(shapes)
```

We then convert the 3-D numpy array to a DataFrame (using `xarray`) for plotting.

```python
da.values = aligned_shapes
df = da.to_unstacked_dataset('xy').to_dataframe().reset_index()
fig, ax = plt.subplots()
sns.lineplot(
    data=df,
    x='x',
    y='y',
    hue='shape',
    style='shape',
    palette='Set2',
    markers=True,
    estimator=None,
    sort=False,
    ax=ax
    )
ax.axis('scaled')
fig.savefig('images/gpa_aligned_triangles.svg')
```

<div align="center">
  <img src="images/gpa_aligned_triangles.svg" />
</div>

The triangles were all the same shape, so they are now perfectly aligned.
