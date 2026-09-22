# Changelog

All notable changes to FedFred will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0a1] - 2026-09-22

First public prerelease of the rebuilt package. The 0.0.x line was a placeholder.

### Added

- Univariate: ARIMA family, GARCH family, Markov switching, threshold and smooth-transition models.
- Multivariate: VAR, VECM, VARMA, factor models; structural identification by recursive, sign, narrative and long-run restrictions.
- State space: Kalman filter and smoother, unobserved-components models, dynamic factor models.
- Bayesian: Minnesota and hierarchical priors, stochastic volatility, Gibbs and particle samplers, marginal likelihood.
- Diagnostics: unit-root and seasonal unit-root tests, nonlinearity tests, long-memory estimators, stability tests.
- Forecast: evaluation, comparison (DM, CW, GW, MCS), combination, calibration.
- Spectral: model-implied and nonparametric spectra, band-pass and trend-cycle filters, Bry-Boschan turning points, MODWT and wavelet coherence.

### Changed

- Requires Python 3.12 or later (PEP 695 type syntax).
- Runtime dependencies are numpy and scipy only; pandas and polars are optional extras.

## [0.0.1] - 2026-05-08

### Added

- Initial file dump.

[Unreleased]: https://github.com/nikhilxsunder/cultivars/compare/v1.0.0a1...HEAD
[1.0.0a1]: https://github.com/nikhilxsunder/cultivars/compare/v0.0.1...v1.0.0a1
[0.0.2]: https://github.com/nikhilxsunder/cultivars/compare/v0.0.1...v0.0.2
