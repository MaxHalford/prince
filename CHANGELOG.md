# Changelog

## 0.18.0

### Bug fixes

- **FAMD: fix categorical variable normalization**. The proportions used to normalize one-hot encoded columns were computed as `n_k / (n * num_cat_vars) * 2` instead of `n_k / n`. This only produced correct results when the dataset had exactly 2 categorical variables. Reported in [#215](https://github.com/MaxHalford/prince/issues/215).

### New features

- **FAMD: `column_correlations` property**. Returns the signed Pearson correlations between standardized quantitative variables and principal components (corresponding to FactoMineR's `quanti.var$coord`). These are the values plotted in a correlation circle. For qualitative variables, η² values are returned (same as `column_coordinates_`).
