# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.7.0

- `prince` is now compatible with `pandas>=1.0`.

### Added

- `PCA` now has an `inverse_transform` function.

### Changed

- `MFA` and `FAMD` should now be able to `fit` and `transform` on categorical variables with different number of distinct values.
- Fixed a bug where `column_correlations` would raise an `AttributeError` in a `FAMD`.
- The output of `column_correlations` is now sorted along the index.
