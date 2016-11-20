===
FAQ
===

**1. After having applied one of Prince's algorithm's on my dataframe, I noticed it's values changed. Why?**

For performance reasons, Prince modifies the provided dataframe inplace. If you don't want the dataframe you provide to be modified, you can use the `copy method <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.copy.html>`_::

    import pandas as pd
    import prince

    df = pd.read_csv('iris.csv')
    pca = prince.PCA(df.copy())

**2. Some of the information on my chart seems to be cut-off, how do I fix this?**

Matplotlib, although being a great library, can be a pain to work with because it's quite low-level  -- things don't get done by magic. One way to fix this is simply to not really on Matplotlib viewer -- the one that appears after ``plt.show()`` -- but rather to directly save the figure with ``bbox_inches='tight'``. Matplotlib has some `documentation <http://matplotlib.org/users/tight_layout_guide.html>`_ covering the issue.::

    import pandas as pd
    import prince

    df = pd.read_csv('.iris.csv')
    pca = prince.PCA(df, n_components=4)

    fig, ax = pca.plot_cumulative_inertia()
    fig.savefig('cumulative_inertia.png', bbox_inches='tight', pad_inches=0.5)
