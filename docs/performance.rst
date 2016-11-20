==========
Perfomance
==========

Prince is made to be used on datasets that fit in memory. Currently `fbpca <https://github.com/facebook/fbpca>`_ is the SVD engine. A PCA on a dataframe of 1M rows and 100 columns took ~6 seconds on an 2013 MacBook Pro (i5, 16G RAM). For out-of-memory SVD and whatnot, check out `Dask <http://dask.pydata.org/en/latest/array-api.html#dask.array.linalg.svd_compressed>`_ and `Spark <https://spark.apache.org/docs/1.2.0/mllib-dimensionality-reduction.html>`_.

Benchmarks incoming!
