==================================
Principal Component Analysis (PCA)
==================================

.. automodule:: prince.pca
    :members:
    :inherited-members:

-------------
Chart gallery
-------------

::

    import pandas as pd
    import prince

    df = pd.read_csv('data/iris.csv')
    pca = prince.PCA(df, n_components=4)

^^^^^^^^^^^^^^^^^^^^^^^^^
Row principal coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    pca.plot_rows(color_by='class', ellipse_fill=True)

.. image:: _static/pca_row_principal_coordinates.png

^^^^^^^^^^^^^^^^^^
Correlation circle
^^^^^^^^^^^^^^^^^^

::

    pca.plot_correlation_circle()

.. image:: _static/pca_correlation_circle.png

^^^^^^^^^^^^^^^^^^
Cumulative inertia
^^^^^^^^^^^^^^^^^^

::

    pca.plot_cumulative_inertia(threshold=0.8)

.. image:: _static/pca_cumulative_inertia.png
