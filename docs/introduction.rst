============
Introduction
============

Factor analysis is a popular method for projecting/representing high-dimensional data on a smaller dimensions. This can be useful for

* visualizing (the data can be projected on a 2 or 3 dimensional chart),
* creating smaller datasets which preserve as much as possible the information contained in original dataset.

Although factor analysis is popular, practitionners tend to mix concepts up -- Principal Component Analysis (PCA) **is not** Singular Value Decomposition (SVD). Moreover, more advanced methods that extend PCA such as Correspondance Analysis and Factor Analysis of Mixed Data (FAMD) are not very well known -- at least outside of French academia.

The Rennes university published `FactoMineR <http://factominer.free.fr/>`_ in 2008; whilst being a library which offers many possibilities, FactoMineR doesn't seem to be actively maintained. What's more, FactoMineR and the underlying SVD operation are written in pure R, which isn't very efficient. In parallel, `Fast Randomized SVD <https://arxiv.org/pdf/1509.00296.pdf>`_ has become an efficient way to obtain eigen{vectors|values} approximations in drastically less time than regular SVD.

The goal with Prince is to provide a user-friendly library for performing all sorts of large-scale factor analysis. Although `Facebook <https://research.facebook.com/blog/fast-randomized-svd/>`_ and then `sklearn <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html>`_ have implemented randomized SVD, it isn't trivial for users to use them, let alone to understand and visualize results in a timely fashion.

Prince builds on top of Randomized SVD engines such as `fbpca <https://github.com/facebook/fbpca>`_ to provide different kinds of algorithms with out-of-the-box charts. The main advantage of using randomized SVD is that the number of eigenvectors that are calculated can be chosen. This is particularly useful because often one only needs the first few eigenvectors to be able to plot relevant information.

-------------------
Implemented methods
-------------------

^^^^^
Basic
^^^^^

Basic methods are to be used when there isn't any intrinsic structure between variables in a dataset (for example a series of questions that have nothing to do with each other).

- [X] `Principal Component Analysis (PCA) <https://www.wikiwand.com/en/Principal_component_analysis>`_ - For continuous variables
- [X] `Correspondence Analysis (CA) <https://www.wikiwand.com/en/Correspondence_analysis>`_ - For two categorical variables (leading to a contingency table)
- [X] `Multiple Correspondence Analysis (MCA) <https://www.wikiwand.com/en/Multiple_correspondence_analysis>`_ - For more than two categorical variables
- [ ] `Factor Analysis of Mixed Data (FAMD) <https://www.wikiwand.com/en/Factor_analysis_of_mixed_data>`_ - For both continuous and categorical variables (incoming)

^^^^^^^^
Advanced
^^^^^^^^

Advanced methods are to be used when variables or individuals are structured in a natural way (for example a survey with questions grouped around topics).

- [ ] `Generalized Procustean Analysis (GPA) <https://www.wikiwand.com/en/Generalized_Procrustes_analysis>`_ - For continuous variables
- [ ] `Multiple Factorial Analysis (MFA) <https://www.wikiwand.com/en/Multiple_factor_analysis>`_ - For both continuous and categorical variables
- [ ] Dual Multiple Factor Analysis - For when the individuals have to be considered in groups and the variables are continuous

----------------------
Delving into the maths
----------------------

Factor analysis is quite a popular topic. A lot of material is available online. The following papers are the ones we recommend. We find them short, thorough and kind to the eyes.

- :download:`Eigenvalues <papers/Eigenvalues.pdf>`
- :download:`Singular Value Decomposition <papers/SVD.pdf>`
- :download:`Principal Component Analysis <papers/PCA.pdf>`
- :download:`Correspondence Analysis <papers/CA.pdf>`
- :download:`Multiple Correspondence Analysis <papers/MCA.pdf>`
- :download:`Global overview <papers/Overview.pdf>`

For math oriented minds, :download:`Halko's paper <papers/Halko.pdf>` is worth knowing about.
