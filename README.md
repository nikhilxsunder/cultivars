<div align="center">
    <img src="https://raw.githubusercontent.com/nikhilxsunder/cultivars/main/assets/exported/cultivars_banner_transparent.png"  alt="cultivars Logo">
</div>

## Research-grade time series econometrics for Python: ARIMA to state space, VAR to structural identification, Bayesian and spectral methods, on one estimation surface.

|                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CI / Quality**     | [![Build](https://github.com/nikhilxsunder/cultivars/actions/workflows/main.yml/badge.svg)](https://github.com/nikhilxsunder/cultivars/actions/workflows/main.yml) [![Analyze](https://github.com/nikhilxsunder/cultivars/actions/workflows/analyze.yml/badge.svg)](https://github.com/nikhilxsunder/cultivars/actions/workflows/analyze.yml) [![Tests](https://github.com/nikhilxsunder/cultivars/actions/workflows/test.yml/badge.svg)](https://github.com/nikhilxsunder/cultivars/actions/workflows/test.yml) [![Docs](https://github.com/nikhilxsunder/cultivars/actions/workflows/docs.yml/badge.svg)](https://nikhilxsunder.github.io/cultivars/stable/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| **Security**         | [![CodeQL](https://github.com/nikhilxsunder/cultivars/actions/workflows/codeql.yml/badge.svg)](https://github.com/nikhilxsunder/cultivars/actions/workflows/codeql.yml) [![Best Practices](https://www.bestpractices.dev/projects/10158/badge)](https://www.bestpractices.dev/projects/10158) [![Socket](https://socket.dev/api/badge/pypi/package/cultivars/1.0.0a2?artifact_id=tar-gz)](https://socket.dev/pypi/package/cultivars/overview/1.0.0a2/tar-gz)                                                                                                                                                                                                                                                                                                                                                                |
| **Coverage**         | [![Coverage](https://codecov.io/gh/nikhilxsunder/cultivars/graph/badge.svg?token=VVEK415DF6)](https://codecov.io/gh/nikhilxsunder/cultivars)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Packaging**        | [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Hatch project](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pypa/hatch/master/docs/assets/badge/v0.json)](https://github.com/pypa/hatch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Distribution**     | [![PyPI](https://img.shields.io/pypi/v/cultivars.svg?include_prereleases)](https://pypi.org/project/cultivars/) [![Python Versions](https://img.shields.io/pypi/pyversions/cultivars.svg)](https://pypi.org/project/cultivars/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **License**          | [![License](https://img.shields.io/pypi/l/cultivars.svg)](https://github.com/nikhilxsunder/cultivars/blob/main/LICENSE)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Usage**            | [![PyPI Downloads](https://static.pepy.tech/badge/cultivars)](https://pepy.tech/projects/cultivars)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Research / Index** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23028870.svg)](https://doi.org/10.5281/zenodo.23028870)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

> **Pre-release.** cultivars is at `1.0.0a2`. The modelling surface is complete and tested; the public API is settling toward `1.0.0` and module paths may still move between alphas. Install with `--pre` and pin the exact version in anything that must not change under you.

## Used by & Featured In

> Note: Listing does not imply endorsement or affiliation.

### Institutions / Organizations

<a href="https://herbert.miami.edu/" title="University of Miami Herbert Business School">
    <img src="https://mbaworldsummit.com/wp-content/uploads/2019/06/Miami-BU-Logo.png"
         alt="University of Miami Herbert Business School"
         height="150">
</a>

## What it is

cultivars is a single, typed, dependency-light library for the econometric time-series toolkit, written so that a research result can be reproduced from the summary it prints. Every model is a class holding a specification; `fit()` (or `identify()`) returns an immutable record that renders the same way everywhere (`print`, a notebook cell, `summary()`), carries its own diagnostics, and refuses to report what its assumptions do not support. Models that need an input the data cannot supply take it as an argument you chose, and the result restates that choice in its own text.

| Area                                             | Modules                                                 | Contents                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------ | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Univariate**                                   | `cultivars.univariate`                                  | AR/ARMA/ARIMA/SARIMAX, GARCH family (GARCH, GJR, EGARCH, FIGARCH, with ARMA means), stochastic volatility, unobserved components, fractional integration, threshold, smooth-transition and Markov-switching autoregressions                                                                        |
| **Multivariate, reduced form**                   | `cultivars.multivariate.reduced_form`                   | VAR/VARX, VECM, VMA, mixed-frequency and panel VARs, closed global VAR, functional and term-structure (Nelson-Siegel) models                                                                                                                                                                       |
| **Multivariate, large-dimensional**              | `cultivars.multivariate.large_dim`                      | Minnesota and hierarchical Bayesian VARs, Gibbs samplers with shrinkage priors, Student-t and stochastic-volatility VARs, penalized and graphical VARs, dynamic factor models, FAVAR, factor stochastic volatility, Diebold-Yilmaz spillovers                                                      |
| **Multivariate, nonlinear and regime-switching** | `cultivars.multivariate.nonlinear`, `.regime_switching` | Threshold, smooth-transition, functional-coefficient, quantile and time-varying-parameter VARs; Markov-switching VAR, SVAR and dynamic factor model                                                                                                                                                |
| **Structural identification**                    | `cultivars.multivariate.structural`                     | Recursive, long-run, AB-model and mixed zero restrictions; sign and narrative sign restrictions; exact set-identification bounds; external instruments; identification by heteroskedasticity, stochastic volatility and non-Gaussianity; FAVAR identification; perturbation-solved DSGE estimation |
| **State space**                                  | `cultivars.state_space`                                 | Linear-Gaussian substrate (Kalman filter, smoothers, simulation smoother), nonlinear filters (extended, unscented, particle), regime-switching state space                                                                                                                                         |
| **Spectral**                                     | `cultivars.spectral`                                    | Periodogram and spectral density, band-pass filters, cycle extraction, frequency-domain causality, wavelets                                                                                                                                                                                        |
| **Bayesian**                                     | `cultivars.bayes`                                       | Priors, chain diagnostics, posterior checks, marginal likelihood, model combination                                                                                                                                                                                                                |
| **Forecasting**                                  | `cultivars.forecast`                                    | Backtesting, scoring rules, calibration, fan charts, conditional forecasts, model confidence sets, forecast comparison tests                                                                                                                                                                       |
| **Diagnostics**                                  | `cultivars.diagnostics`                                 | Unit-root tests (ADF, DF-GLS, KPSS, Phillips-Perron, Ng-Perron), cointegration, structural breaks, stability, seasonality, nonlinearity, long memory                                                                                                                                               |

Dependencies are `numpy`, `scipy` and [`fedfred`](https://github.com/nikhilxsunder/fedfred) (FRED data access). `pandas` and `polars` are optional and only used for the `to_pandas()` / `to_polars()` conveniences on results.

## Installation

Requires Python 3.12 or later.

```bash
pip install --pre cultivars
```

With the DataFrame extras:

```bash
pip install --pre "cultivars[pandas]"      # or [polars]
```

With `uv`:

```bash
uv add --prerelease allow cultivars
```

For a reproducible research environment, pin the alpha explicitly: `cultivars==1.0.0a2`. A conda-forge package will follow the first stable release.

## Quick start

### A univariate model

```python
import numpy as np
from cultivars.univariate.box_jenkins import ARIMA

rng = np.random.default_rng(0)
y = np.zeros(300)
for t in range(1, 300):
    y[t] = 0.6 * y[t - 1] + rng.standard_normal()

res = ARIMA(y, order=(1, 0, 0)).fit()
print(res)
```

```text
                             ARIMA(1, 0, 0) Results
================================================================================
Model:                  ARIMA(1, 0, 0)  Log-likelihood:               -430.692
Trend:                               c  AIC:                           867.383
Exog regressors:                     0  BIC:                           878.495
Observations:                      300  HQIC:                          871.830
--------------------------------------------------------------------------------
              coef
const      -0.0961
ar.L1       0.6245
sigma2      1.0323
================================================================================
Stationary: True   max |AR root| = 0.6245
```

Every result is a frozen dataclass: `res.params`, `res.resid`, `res.information_criteria`, `res.is_stationary` are attributes, `res.compare(other)` and `res.likelihood_ratio_test(other)` rank and test nested specifications, and `res.to_pandas()` hands the per-observation series to a DataFrame.

### A VAR, its diagnostics, and a structural view

```python
from cultivars.multivariate.reduced_form.vector_autoregression import VAR
from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

Y = np.zeros((300, 2))
for t in range(1, 300):
    Y[t] = np.array([[0.5, 0.1], [0.0, 0.4]]) @ Y[t - 1] + rng.standard_normal(2)

var = VAR(Y, order=1, names=("gdp", "infl")).fit()
var.forecast(3)                              # (3, 2) point forecasts
var.irf(20)                                  # orthogonalized responses by default
print(var.granger_causality("infl", "gdp"))  # a Wald test with its verdict
```

```text
                                   Wald Test
================================================================================
Verdict at 5%:                    keep
--------------------------------------------------------------------------------
null                                statistic   df   p-value
infl does not Granger-cause gdp        2.6420    1    0.1041
================================================================================
```

Identification is a separate step that takes the fitted reduced form and the restriction you are prepared to defend:

```python
svar = RecursiveSVAR(var, order=("gdp", "infl")).identify()
svar.impact                 # (k, k) impact matrix, lower triangular in the declared order
svar.irf(20)                # structural responses
svar.fevd(20)               # variance shares, summing to one across identified shocks
svar.structural_shocks()    # recovered unit-variance shocks
print(svar.restriction)
```

```text
Recursive ordering gdp -> infl: each variable responds on impact only to shocks at or before its own position. Permuting the ordering changes the answer; that is the identifying assumption, not a numerical artifact.
```

The same `VAR` result feeds every scheme in `cultivars.multivariate.structural`: sign restrictions return the accepted set rather than a point, external instruments return exactly the one column they identify, and the statistical schemes test their own identifying condition and print the verdict.

### Volatility

```python
from cultivars.univariate.conditional_variance import GARCH

r = rng.standard_normal(500) * np.sqrt(np.r_[np.ones(250), 3 * np.ones(250)])
garch = GARCH(r, p=1, q=1).fit()
print(garch)
```

```text
                              GARCH(1, 1) Results
================================================================================
Model:                     GARCH(1, 1)  Log-likelihood:               -863.345
Mean:                         constant  AIC:                          1734.690
AR lags:                             0  BIC:                          1751.548
Observations:                      500  HQIC:                         1741.305
--------------------------------------------------------------------------------
                coef
const        -0.0239
omega         0.0171
alpha[1]      0.0627
beta[1]       0.9359
================================================================================
Persistence: 0.9985   Covariance stationary: True   Half-life: 465.5
Unconditional variance: 11.4758
```

### A diagnostic

```python
from cultivars.diagnostics.unit_roots import adf

walk = np.cumsum(rng.standard_normal(300))
print(adf(walk, trend="c"))
```

```text
                                    ADF Test
================================================================================
Null:                        unit root  Trend:                               c
Lags:                                0  Method:                   aic (max 15)
Observations:                      299  Verdict at 5%:          keep unit root
--------------------------------------------------------------------------------
statistic       value   p-value       1%       5%      10%
ADF           -0.7203    0.8414   -3.452   -2.871   -2.572
================================================================================
Rejection lies in the lower tail.
```

## Design

Three commitments run through the package and are worth knowing before you read any docstring.

**Estimation and identification are separate acts.** A reduced form is fitted once and reports only what the data determine. Anything that needs an assumption (a causal ordering, a sign pattern, an instrument, a prior) is constructed from that result, declares what it adds, and restates the declaration in its summary, so an impulse response can always be traced to the assumption that signed it.

**Results refuse what they cannot support.** A partially identified scheme carries only the columns it identified. A set-identified scheme returns the set. A mixture model refuses the regime-count likelihood-ratio test its likelihood cannot answer. A long-run restriction refuses a unit root. A particle-filter likelihood says it is an estimate. Where a method is a known approximation, the summary names it.

**Failures say what kind of fix is needed.** Every exception the package raises derives from `cultivars.exceptions.CultivarsError` and from one of three subclasses: `DimensionError` (reshape the data), `SpecificationError` (change the choice), `NumericalError` (look at the numbers). Each also derives from the matching builtin, so code written against `ValueError` and `RuntimeError` keeps working.

Packages re-export modules rather than classes; import a model from the module that defines it, as in the examples above.

## Documentation

- Stable: <https://nikhilxsunder.github.io/cultivars/stable/>
- Development: <https://nikhilxsunder.github.io/cultivars/dev/>

The API reference is generated from the docstrings, every one of which carries a runnable, deterministic example; the examples double as the doctest suite.

## Development

```bash
git clone https://github.com/nikhilxsunder/cultivars.git
cd cultivars
uv sync --all-groups
uv run pytest                                   # unit tests
uv run pytest --doctest-modules src/cultivars   # docstring examples
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run make -C docs html                        # documentation
```

Contributions are welcome; see [CONTRIBUTING.md](https://github.com/nikhilxsunder/cultivars/blob/main/CONTRIBUTING.md) for the conventions (Google-style docstrings with doctests, strict typing, no silent failure) and [SECURITY.md](https://github.com/nikhilxsunder/cultivars/blob/main/SECURITY.md) for reporting vulnerabilities.

## Citation

If cultivars contributes to published work, cite the concept DOI, which resolves to the latest release:

```bibtex
@software{sunder_cultivars,
  author  = {Sunder, Nikhil},
  title   = {cultivars: Research-grade time series econometrics for Python},
  year    = {2026},
  doi     = {10.5281/zenodo.23028870},
  url     = {https://github.com/nikhilxsunder/cultivars}
}
```

## License

MIT. See [LICENSE](https://github.com/nikhilxsunder/cultivars/blob/main/LICENSE).
