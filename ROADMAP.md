# Roadmap

This document states what cultivars intends to do, and what it does not intend to do, over the next year. Releases are gated on exit criteria, not dates. Issue-level tracking lives in the [GitHub milestones](https://github.com/nikhilxsunder/cultivars/milestones); this file is the summary those milestones roll up to.

Last updated: 2026-10-01

## Status

| Release | Theme                                                                | State               |
| ------- | -------------------------------------------------------------------- | ------------------- |
| 1.0.0a1 | First public prerelease: the full model surface                      | Released 2026-09-22 |
| 1.0.0b1 | Validated core: tests, reference fixtures, doctests, narrative docs  | In progress         |
| 1.0.0   | Stable: API guarantee on the validated core                          | Planned             |
| 1.1.0b1 | Pluggable compute engines: MLX and CuPy behind a versioned contract  | Planned             |
| 1.1.0   | Compute engines stable                                               | Planned             |
| 1.2.0   | Model additions: local projections and the `experimental` subpackage | Planned             |

## Principles

1. **Validation before breadth.** No new model classes land until the 1.0 core is validated.
2. **Correctness by comparison.** A model is stable only when it reproduces a reference implementation or a published table within a documented tolerance.
3. **Two runtime dependencies.** numpy and scipy. Everything else, including accelerated engines and dataframe support, is optional.
4. **Scoped stability.** Semantic Versioning guarantees apply to the stable core. Everything else is marked experimental and may change in any minor release.
5. **Criteria, not dates.** A release ships when its exit criteria are met.

## 1.0.0b1: validated core

The alpha shipped the model surface. The beta is the evidence that the core of it is correct.

**Stable core.** VAR, VARX, VECM, RecursiveSVAR, ShortRunSVAR, LongRunSVAR, SignRestrictedSVAR, ProxySVAR, BVAR, HierarchicalBVAR, BVARSV, DFM, LinearGaussianSSM, UnobservedComponents, MSAR, SETAR, TVAR, TVPVARSV, MarkovSwitchingVAR, RegimeSwitchingLinearSSM, plus the `diagnostics` tests and the `forecast` comparison functions. Every other public class carries an `Experimental` admonition and is excluded from the stability guarantee.

**Work**

- Test tree: `tests/` mirrors `src/`, with unit tests and Hypothesis property tests on the `_core` primitives (companion roots, difference and undifference round trips, validator failure paths).
- Reference fixtures: pinned outputs so comparisons run in CI without the external dependency.

  | Models                                                              | Oracle                            |
  | ------------------------------------------------------------------- | --------------------------------- |
  | VAR, VARX, VECM, MSAR, UnobservedComponents, LinearGaussianSSM, DFM | statsmodels                       |
  | BVAR, HierarchicalBVAR                                              | R `BVAR` on FRED-QD               |
  | TVPVARSV                                                            | R `bvarsv` (`usmacro`)            |
  | SETAR, TVAR                                                         | R `tsDyn` (`lynx` for SETAR)      |
  | RecursiveSVAR, ShortRunSVAR, LongRunSVAR                            | R `svars` (`USA`)                 |
  | SignRestrictedSVAR                                                  | R `bsvarSIGNs`                    |
  | ProxySVAR                                                           | Mertens and Ravn published tables |
  | RegimeSwitchingLinearSSM                                            | Kim and Nelson published tables   |
  | BVARSV, MarkovSwitchingVAR                                          | To be selected                    |

- Docstrings: every core class has Args, Example, and References.
- Doctests run in CI with `--doctest-modules`; the Sphinx build runs with `-W`.
- Docs: install, quickstart, one "which model when" page per subpackage, and three notebooks executed in CI.
- Housekeeping: README, CHANGELOG, and CONTRIBUTING consistent with the package as shipped.

**Exit criteria.** All of the above, with coverage at the CONTRIBUTING floor: 90% on the core and 100% branch coverage on the `_core` primitives.

## 1.0.0: stable

- A beta period with no API changes to the stable core.
- Classifier set to `Production/Stable`; documentation version banner enabled.
- Release tagged, signed, and published by `release.yml`; conda-forge feedstock submitted.

From 1.0.0 onward, a breaking change to the stable core requires a major version.

## 1.1.0b1: compute engines

GPU and multi-backend execution behind a versioned engine contract. The NumPy reference engine stays in core as the default and the fallback; accelerated engines ship as optional plugin distributions and are selected explicitly. The public model API does not change.

**Sequence**

1. Engine contract in core: `Engine` protocol, capability flags, entry-point registry, and an integer contract version that the registry enforces. The existing NumPy engine moves behind it with numerically identical results.
2. Public conformance suite, run against the NumPy reference with documented per-dtype tolerances.
3. MLX engine for Apple Silicon, with float64 and linear algebra routed by capability flags.
4. CuPy engine: one code path for CUDA and ROCm. ROCm is best-effort until hardware CI exists.
5. Benchmarks with explicit device synchronization, and docs: selection guide, capability matrix, plugin author guide.

**Contract guarantees**

- No implicit precision downcast. float64 input on an engine without float64 raises unless the caller opts in.
- Unsupported operations fall back per operation to the reference engine and log once; strict mode raises instead.
- Not installed, failed import, and no device available are three distinct errors.

## 1.1.0: compute engines stable

- A beta period with no change to the engine contract version.
- Plugin distributions published alongside core, each passing the conformance suite.

## 1.2.0: model additions

**Local projections** in `multivariate`: Jordà (2005), lag-augmented LP, LP-IV, smooth LP, and Bayesian LP, validated against R `lpirfs`. This is the largest gap in the mainstream surface.

**Generalized impulse responses** as a shared utility for the nonlinear models (TVAR, STVAR).

**`cultivars.experimental`**, a subpackage for recent models that are not yet packaged anywhere:

- Not imported by the top-level package; importing it emits a one-time `ExperimentalWarning`.
- Every class carries structured metadata: paper, oracle, and what has not been verified.
- Same coverage floor and the same numpy and scipy contract as core.
- Promotion: once the replication fixture lands and the API has been stable for two minor releases, the class moves to its permanent subpackage. Four minor releases without promotion and it is removed.

First candidates, in order: M\*-BVAR (Hong, Kang, and Kim 2026), shadow-rate BVAR (Carriero, Clark, Marcellino, and Mertens 2025), multi-frequency echo state networks (Ballarin et al. 2024), and Gaussian process VARs (Hauzenberger, Huber, Marcellino, and Petz 2025).

## Under consideration

Not scheduled. Each needs a concrete case before it gets a milestone.

- Additional GPU kernel dialects (`numba.cuda.jit`, Triton), per kernel and only with a profile showing the gain.
- Partial identification with non-centred stochastic volatility, as an extension of `StochasticVolatilitySVAR`.
- Spectral factor models and the extended Wold decomposition in `spectral`.
- Hemisphere neural networks, conditional on the `ARNN` trainer accepting an architecture object.

## Not planned

- Deep-learning frameworks (PyTorch, JAX, TensorFlow) as dependencies of core, and models that require them: foundation models and deep state-space models.
- BART and other tree-ensemble VARs. A numpy sampler stack for these is a separate project.
- Reimplementations of models that already have a maintained Python package.
- Multi-GPU and distributed execution.

## How this roadmap changes

The file is revised at each release and whenever a milestone's scope changes. Proposals for new models or scope changes go through a GitHub issue; see [CONTRIBUTING.md](CONTRIBUTING.md) for the testing and reference-comparison requirements a new estimator must meet.
