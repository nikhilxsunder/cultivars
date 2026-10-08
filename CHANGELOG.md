# Changelog

All notable changes to cultivars will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0.0, pre-releases may move public names between alphas; such moves are
listed under Changed.

## [Unreleased]

### Fixed

- `ARMA` validates the two-element order before unpacking, so wrong-length
  orders raise `SpecificationError` with the expected `p` and `q` terms.

## [1.0.0a2] - 2026-10-08

Documentation release. Every public module, class, method and dataclass field
now carries a Google-style docstring with a runnable, deterministic example;
the examples are the doctest suite. The pass also surfaced several hundred
findings, which are tracked as GitHub issues against the 1.0.0 milestone.

### Added

- API reference generated from the docstrings, with a sphinx-gallery examples
  section, source links, and a version switcher that works on the published
  site and in a local preview (`make -C docs serve`).
- `README.md` with installation, quick-start examples and the package layout.
- Module docstrings for every package and subpackage, stating what each
  family of models commits to and where its helpers live.
- `cultivars.exceptions` documents the full hierarchy with the handler
  patterns each class supports.

### Changed

- Documentation builds no longer fail on warnings or on a failing gallery
  example; `make -C docs strict` (and the workflow's `strict` input) restore
  warnings-as-errors.
- `VARResult` docstring examples and cross-references now use module paths
  (`cultivars.multivariate.reduced_form.vector_autoregression.VAR`); the
  subpackage `__init__` modules re-export modules only, not classes.

### Fixed

- `VARResult._from_fit` dropped the `posterior` and `prior_label` fields when
  building a result from a Bayesian (Minnesota-prior) fit, so a shrinkage
  estimate reported as if it were OLS.
- The `exceptions` module described `StabilityWarning` as emitted by the
  fitting layers; no layer emits it yet, and the docstring now says what the
  class is for rather than what does not happen.
- Stale cross-references to classes that do not exist (`SignSVAR`,
  `NarrativeSignSVAR`, `DSGESpecification`) and to the old `cultivars.var.*`
  paths, across the multivariate packages.

## [1.0.0a1] - 2026-09-22

First public pre-release of the rebuilt package. The 0.0.x line was a
placeholder.

### Added

- Univariate: ARIMA family, GARCH family, stochastic volatility, unobserved
  components, fractional integration, Markov-switching, threshold and
  smooth-transition models.
- Multivariate, reduced form: VAR and VARX, VECM, VMA, mixed-frequency and
  panel VARs, closed global VAR, functional and Nelson-Siegel term-structure
  models.
- Multivariate, large-dimensional: Minnesota and hierarchical Bayesian VARs,
  Gibbs samplers with shrinkage priors, Student-t and stochastic-volatility
  VARs, penalized and graphical VARs, dynamic factor models, FAVAR, factor
  stochastic volatility, Diebold-Yilmaz spillovers.
- Multivariate, nonlinear and regime-switching: threshold, smooth-transition,
  functional-coefficient, quantile and time-varying-parameter VARs;
  Markov-switching VAR, SVAR and dynamic factor model.
- Structural identification: recursive, long-run, AB-model and mixed zero
  restrictions; sign and narrative sign restrictions; exact set-identification
  bounds; external instruments; identification by heteroskedasticity,
  stochastic volatility and non-Gaussianity; FAVAR identification;
  perturbation-solved DSGE estimation by Kalman, extended, unscented and
  particle filters with particle marginal Metropolis-Hastings.
- State space: linear-Gaussian substrate with Kalman filter, smoothers and
  simulation smoother; extended, unscented and particle filters;
  regime-switching state space.
- Bayesian: priors, chain diagnostics, posterior checks, marginal likelihood,
  model combination.
- Diagnostics: ADF, DF-GLS, KPSS, Phillips-Perron and Ng-Perron unit-root
  tests, cointegration, structural breaks, stability, seasonality,
  nonlinearity and long-memory tests.
- Forecast: backtesting, scoring rules, calibration, fan charts, conditional
  forecasts, Diebold-Mariano, Clark-West and Giacomini-White comparisons,
  model confidence sets, combination.
- Spectral: periodogram and spectral density, band-pass and trend-cycle
  filters, cycle extraction and Bry-Boschan turning points, frequency-domain
  causality, MODWT and wavelet coherence.
- Typed throughout (`py.typed`); results are frozen dataclasses with a
  uniform `summary()` and `to_pandas()` / `to_polars()` conveniences.

### Changed

- Requires Python 3.12 or later (PEP 695 type syntax).
- Runtime dependencies are `numpy`, `scipy` and `fedfred`; `pandas` and
  `polars` are optional extras.

## [0.0.1] - 2026-05-08

### Added

- Initial file dump.

[Unreleased]: https://github.com/nikhilxsunder/cultivars/compare/v1.0.0a2...HEAD
[1.0.0a2]: https://github.com/nikhilxsunder/cultivars/compare/v1.0.0a1...v1.0.0a2
[1.0.0a1]: https://github.com/nikhilxsunder/cultivars/compare/v0.0.1...v1.0.0a1
[0.0.1]: https://github.com/nikhilxsunder/cultivars/releases/tag/v0.0.1
