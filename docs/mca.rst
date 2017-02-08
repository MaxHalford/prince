======================================
Multiple Correspondance Analysis (MCA)
======================================

.. automodule:: prince.mca
    :members:
    :inherited-members:

-------------
Chart gallery
-------------

::

    import pandas as pd
    import prince

    df = pd.read_csv('data/ogm.csv')
    mca = prince.MCA(df, n_components=-1)

^^^^^^^^^^^^^^^^^^^^^^^^^
Row principal coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    mca.plot_rows(show_points=True, show_labels=False, color_by='Position Al A', ellipse_fill=True)

.. image:: _static/mca_row_principal_coordinates.png

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Row and column principal coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    mca.plot_rows_columns()

.. image:: _static/mca_row_column_principal_coordinates.png

^^^^^^^^^^^^^^^^^^^
Relationship square
^^^^^^^^^^^^^^^^^^^

::

    mca.plot_relationship_square()

.. image:: _static/mca_relationship_square.png

^^^^^^^^^^^^^^^^^^
Cumulative inertia
^^^^^^^^^^^^^^^^^^

::

    mca.plot_cumulative_inertia(threshold=0.8)

.. image:: _static/mca_cumulative_inertia.png
