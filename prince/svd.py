import fbpca


class SVD():

    """This object contains the results from a Singular Value Decomposition. The results are created
    when the object is instanciated."""

    U = None
    s = None
    V = None

    def __init__(self, X, **kwargs):
        """Perform the Singular Value Decomposition (SVD) of a matrix `X` of shape `(n, p)`.

        Args:
            X (matrix): The matrix on which to perform the SVD. `X` can be a `pandas.DataFrame`,
                however the SVD will be faster with a pure `numpy` matrix which can be extracted
                from a `pandas.DataFrame` thanks to the `values` property.
            kwargs: see http://fbpca.readthedocs.io/en/latest/#fbpca.pca.

        Returns:
            matrix: The left eigenvectors of shape `(n, k)`, usually denoted `U`.
            array: The singular values (square roots of the eigenvalues) of shape `(k,)`, usually
                denoted `s`.
            matrix: The right eigenvectors of shape `(k, p)`, usually denoted `V`.
        """
        self.U, self.s, self.V = fbpca.pca(X, **kwargs)
