# Changelog

## 0.20.0 — unreleased

### Bug fixes

- **MCA: incorrect Greenacre correction with subset MCA**. When `one_hot_columns_to_drop` was set, the closed-form Benzécri/Greenacre formula was still applied, but it assumes uniform row sums in the indicator matrix — which subsetting violates. The corrected path uses Greenacre's subset-MCA adjustment (CA in Practice, ch. 21), ported from R's `ca::mjca(lambda='adjusted', subsetcat=...)`. The non-subset path is unchanged. Fixes [#206](https://github.com/MaxHalford/prince/issues/206).
- **PCA: supplementary column coordinates were missing a factor of √n**. Coordinates exceeded [-1, 1] instead of being correlations with the principal components. The standardization of `X_sup` and `_column_dist` also ignored `sample_weight`, making them inconsistent with the active-variable treatment. The fix now matches FactoMineR's `pca$quanti.sup$coord` and propagates to MFA supplementary groups.
- **SVD: `random_state` was ignored with `engine='fbpca'`**. Output was deterministic regardless of the user-supplied seed. The seed is now passed through to fbpca, and the fbpca path no longer mutates the global numpy RNG state.

### New features

- **MFA: categorical groups**. Categorical groups are now fitted with MCA (indicator columns centered and divided by √(p_j·(1-p_j)), with column weight (1-p_j)/(λ₁·Q)). This allows numeric and categorical groups to be mixed in a single MFA. Adds `prince.datasets.load_poison()`. Closes [#231](https://github.com/MaxHalford/prince/issues/231).
- **MCA: faster `fit`**. The indicator matrix is now built directly as a scipy CSC sparse matrix (factorize columns, build COO from offsets) instead of going through a dense intermediate, with a single densification at the end. Materially faster on wide categorical datasets.
- **`prince.__version__`**. The package version is now exposed as `prince.__version__`.

## 0.19.0 — 2026-05-05

### Bug fixes

- **MFA: `column_cosine_similarities_` was not callable**. Calling `mfa.column_cosine_similarities_(X)` raised `TypeError: 'DataFrame' object is not callable` because a method override was trying to call the parent class's `@property`. The override was removed — the inherited property works correctly. Reported in [#218](https://github.com/MaxHalford/prince/issues/218).
- **MFA: `partial_row_coordinates` broken with supplementary groups**. The method crashed with a shape mismatch when supplementary groups were specified, because column weights and the data matrix had incompatible dimensions.
- **MFA: `partial_row_coordinates` used wrong scaling with supplementary rows**. The method recomputed mean/std from the input data (including supplementary rows) instead of using the fitted scaler parameters, producing incorrect results.

### New features

- **MFA: group-level results**. Added `group_coordinates_`, `group_contributions_`, and `group_cosine_similarities_` properties. These summarize how each group of variables relates to the global MFA components, corresponding to FactoMineR's `result$group$coord`, `result$group$contrib`, and `result$group$cos2`. Requested in [#217](https://github.com/MaxHalford/prince/issues/217).
- **MFA: partial axes results**. Added `partial_correlations_` and `partial_contributions_` properties. These describe how each group's own PCA axes relate to the global MFA axes, corresponding to FactoMineR's `result$partial.axes$cor` and `result$partial.axes$contrib`. Requested in [#217](https://github.com/MaxHalford/prince/issues/217).
- **MFA: partial axes correlation plot**. Added `plot_partial` method. Produces a correlation circle showing how each group's PCA axes project onto the global MFA plane, corresponding to FactoMineR's `plot.MFA(res, choix="axes")`.

## 0.18.0 — 2026-05-04

### Bug fixes

- **FAMD: fix categorical variable normalization**. The proportions used to normalize one-hot encoded columns were computed as `n_k / (n * num_cat_vars) * 2` instead of `n_k / n`. This only produced correct results when the dataset had exactly 2 categorical variables. Reported in [#215](https://github.com/MaxHalford/prince/issues/215).

### New features

- **FAMD: `column_correlations` property**. Returns the signed Pearson correlations between standardized quantitative variables and principal components (corresponding to FactoMineR's `quanti.var$coord`). These are the values plotted in a correlation circle. For qualitative variables, η² values are returned (same as `column_coordinates_`).
