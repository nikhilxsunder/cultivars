# Contributing to cultivars

Thank you for your interest in contributing to cultivars! cultivars is a research-grade Python SDK for autoregressive time-series modeling, and we welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code. This document provides the guidelines and instructions to get you started.

Before writing code, please read [`architecture.md`](.docs/architecture.md). It is binding for all contributions — it specifies the layering, the three-object discipline, the naming conventions, and the locked decisions that a PR may not silently violate. A change that conflicts with a locked decision requires a PR that updates `architecture.md` and a version bump.

## Table of Contents

- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
- [Development Setup](#development-setup)
- [Architecture Compliance](#architecture-compliance)
- [Coding Standards](#coding-standards)
- [Static Analysis](#static-analysis)
- [Testing](#testing)
  - [Coverage Requirements](#coverage-requirements)
  - [Reference-Comparison Testing](#reference-comparison-testing)
  - [Property-Based Testing with Hypothesis](#property-based-testing-with-hypothesis)
  - [A Note on Assertions](#a-note-on-assertions)
- [Submitting Code Changes](#submitting-code-changes)
- [Pull Request Process](#pull-request-process)
- [Continuous Integration](#continuous-integration)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [License](#license)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)
- [Code of Conduct](#code-of-conduct)
- [Governance Model](#governance-model)
- [Contact](#contact)

## How to Contribute

### Reporting Issues

Before opening a new issue:

- Search the issue tracker to verify the issue hasn't already been reported.
- Use our issue templates where available for bugs, features, or documentation.
- Provide a clear and descriptive title.
- Include detailed information:
  - Steps to reproduce, ideally a minimal runnable snippet.
  - Expected behavior vs. actual behavior.
  - Environment details (OS, Python version, `numpy`/`scipy` versions, cultivars version).
  - Relevant tracebacks or error messages. For numerical discrepancies, include the reference implementation and values you compared against.
- Tag the issue appropriately (bug, enhancement, documentation, etc.).

## Development Setup

cultivars uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

1. **Prerequisites**
   - Python 3.11+ (the project floor; see `.python-version`)
   - `uv`
   - Git

2. **Clone and sync**

   ```bash
   git clone https://github.com/nikhilxsunder/cultivars.git
   cd cultivars

   # Create the virtual environment and install runtime + dev dependencies
   uv sync

   # Install the pre-commit hooks
   uv run pre-commit install
   ```

   `uv sync` reads the pinned interpreter from `.python-version` and resolves the environment from `pyproject.toml` / `uv.lock`. Develop on the project floor (3.11) so you catch accidental use of newer-version features; CI runs the full matrix upward.

3. **Optional extras**

   Some capabilities live behind optional extras. Install what a change touches:

   ```bash
   uv sync --extra jax      # JAX-accelerated paths
   uv sync --extra plot     # matplotlib plotting
   uv sync --extra all      # everything
   ```

   Only the optional data-loader extra needs credentials (a FRED API key) — and only when running its integration tests. The core library performs no network I/O and requires no configuration.

## Architecture Compliance

cultivars is defensible because it is disciplined. A PR is expected to respect the following, all of which are enforced in review and most in CI:

- **Layer boundaries.** The four layers (state-space substrate → reduced-form estimators → model classes → composition) depend only downward. No upward or backward imports. `import cultivars.state_space` must not pull in higher layers. This is enforced by `import-linter`.
- **Three-object discipline.** Every model decomposes into an immutable `Spec`, a transient `Estimator`, and a frozen `Result`. One `Spec`, one `Estimator`, one `Result` per file.
- **Strategy-pattern composition.** Identification schemes, priors, regimes, and estimation backends are objects passed to a model — never new subclasses. No `BVAR_Minnesota_SV_Sign` class explosion.
- **Substrate purity.** `cultivars/core/` and `cultivars/state_space/` import only `numpy` and `scipy`. Heavy or optional dependencies are lazy-loaded behind extras and `TYPE_CHECKING`.
- **File discipline.** Files stay under roughly 400 lines except where the mathematics genuinely requires more. `tests/` mirrors the package tree exactly.
- **Module header.** Every `.py` file begins with the MIT header (copyright 2026 Nikhil Sunder, SPDX identifier) followed by the module docstring and `from __future__ import annotations`. See the template in `architecture.md`.

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/); style is enforced by `ruff format` and `ruff check`.
- Provide complete [type hints](https://peps.python.org/pep-0484/) on every public parameter and return value. The public surface must pass `mypy --strict`. Respect superclass contracts, use proper overloads, and do not weaken `Optional` defaults.
- Write **Google-style docstrings** with doctest examples where they clarify behavior. Document parameters, return values, and every exception raised. Mathematical content uses inline LaTeX in the description.
- Use meaningful names; keep functions focused. Avoid unnecessary complexity and deep nesting.
- Practice defensive programming: explicit key/shape/finiteness checks, specific exception classes from `cultivars.exceptions`, and comprehensive error messages. No silent failure, no bare `except:`.
- Thread randomness through explicit `np.random.Generator` instances routed via `cultivars._internal.random`. Module-level `np.random.seed(...)` is banned (enforced by `ruff NPY002`).

## Static Analysis

All code is checked with the following before merge. Run them locally before opening a PR:

```bash
# Lint (includes flake8-bandit "S" security rules) and format check
uv run ruff check src/cultivars/ tests/
uv run ruff format --check src/cultivars/ tests/

# Strict static type checking on the public surface
uv run mypy src/cultivars/

# Layer-dependency enforcement
uv run import-linter --config pyproject.toml
```

Security-focused static analysis is covered by ruff's `S` (flake8-bandit) ruleset plus CodeQL in CI. These checks are automated through pre-commit hooks (for developers), GitHub Actions (for all PRs and releases), and required status checks — a PR cannot merge if static analysis fails. Any suppressed warning must carry an inline comment justifying the suppression.

## Testing

All new functionality and every bug fix must be accompanied by tests. Tests live under `tests/`, mirroring the package tree exactly — `cultivars/core/lag.py` is tested by `tests/core/test_lag.py`. Test both success and failure paths, and include edge and boundary conditions. Run the suite with:

```bash
uv run pytest
```

### Coverage Requirements

- The project floor is **90% coverage** (`--cov-fail-under=90`); PRs that drop below it will not merge.
- The `cultivars/core/` primitives (`lag`, `companion`, `stability`, `transforms`) require **100% branch coverage** — a defect there is a silent defect in every model that composes through them.

```bash
uv run pytest --cov=cultivars --cov-report=term-missing
```

### Reference-Comparison Testing

Correctness-by-comparison is cultivars' central quality strategy, not an optional extra. Where a model has an authoritative reference implementation, the test suite must validate against it numerically:

- Reduced-form estimators against `statsmodels` on identical data.
- Bayesian estimators against R's `BVAR` (or the relevant reference) to a stated decimal tolerance.
- Analytic quantities (IRFs of a known VAR(1), companion eigenvalues vs. the roots of the AR polynomial) against closed-form values.

Pin the reference values as fixtures so the comparison runs in CI without the external dependency at test time. When you add a new estimator, add its reference-comparison test in the same PR.

### Property-Based Testing with Hypothesis

cultivars uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based tests that assert invariants across generated inputs rather than fixed cases. These are especially valuable for numerical primitives. Examples of properties worth asserting:

- `undifference(difference(x, d), initials, d=d)` reconstructs `x`.
- `standardize` followed by its inverse is the identity.
- The companion eigenvalues equal the reciprocals of the AR polynomial roots.
- A `Spec` that constructs successfully round-trips through serialization unchanged.

```bash
# Run property-based tests with detailed statistics
uv run pytest tests/ -k property --hypothesis-show-statistics
```

### A Note on Assertions

cultivars does **not** use `assert` for runtime input validation. Because `python -O` strips assertions, validation that guards user input must raise a specific exception from `cultivars.exceptions` instead. `assert` is reserved for the test suite, where it expresses expected outcomes. This is the opposite of relying on assertions in library code, and it is deliberate: a validation check that can be optimized away is not a validation check.

## Submitting Code Changes

1. **Fork and clone** the repository.
2. **Create a branch** from `main` with a descriptive, prefixed name (`feature/`, `fix/`, `docs/`), e.g. `feature/sign-restricted-svar`.
3. **Make focused changes** following the standards above, with clear, logical commits.
4. **Sign off your commits (DCO).** Every commit must carry a `Signed-off-by` line certifying you have the right to contribute the code under the project license:

   ```bash
   git commit -s -m "Add sign-restricted SVAR identification"
   ```

   This appends `Signed-off-by: Your Name <your.email@example.com>`, indicating agreement to the [Developer Certificate of Origin](https://github.com/nikhilxsunder/cultivars/blob/main/DCO.md). PRs with unsigned-off commits will not be merged.

   Separately, **cryptographically signed commits are encouraged.** DCO sign-off (`-s`) is a legal certification; a verified signature (`-S`, or a configured `commit.gpgsign`) is what earns GitHub's "Verified" badge. If you have SSH or GPG commit signing configured, leave it on.

5. **Submit a pull request** against `main`, ensuring the full suite passes locally first.

## Pull Request Process

1. **Submission** — open the PR against `main`, fill out the template completely, link related issues (e.g. "Fixes #42"), and keep it focused on a single objective.
2. **Review** — at least one maintainer reviews. Expect initial feedback within one to two weeks. Address requested changes and push updates. Once approved, a maintainer merges.
3. **After merging** — your contribution ships in the next release and you are added to the contributors list.

## Continuous Integration

GitHub Actions runs on every push and PR. The pipeline blocks merge on any of:

- `mypy --strict` failure on the public surface.
- `ruff check` / `ruff format --check` failure.
- `pytest --cov=cultivars --cov-fail-under=90` failure.
- `import-linter` failure on the layer dependencies.
- The import-budget benchmark exceeding the target (`import cultivars` under 200 ms on a 2024 M-series Mac).
- Benchmark regressions beyond the tracked threshold (nightly; advisory before v0.5, blocking after).

You can run the gating checks locally with the commands in [Static Analysis](#static-analysis) and [Testing](#testing).

## Documentation

- Update the relevant documentation whenever you change behavior.
- Document all public APIs with Google-style docstrings and runnable examples for non-trivial functionality.
- The docs site is built with Sphinx (`sphinx-design`, `myst-parser`) and hosted on Read the Docs; mathematical content renders via MathJax.
- For markdown files: consistent heading hierarchy, appropriate links, and fenced code blocks with a language specified.

## Release Process

cultivars follows [Semantic Versioning](https://semver.org/): MAJOR for incompatible API changes, MINOR for backward-compatible additions, PATCH for backward-compatible fixes. Note that cultivars is pre-1.0 — the decisions in `architecture.md` §6 are locked, but the wider public API may change between minor releases until 1.0.

Release procedure:

1. Open a PR with the version bump and `CHANGELOG` updates.
2. Label it `release-candidate` to trigger the full dynamic-analysis and benchmark suite.
3. Review the results, including property-based and reference-comparison tests.
4. After merge, tag the release with a **signed** tag (`git tag -s vX.Y.Z`). A GitHub Actions workflow then publishes to PyPI via **Trusted Publishing (OIDC)** with PEP 740 digital attestations — no long-lived upload tokens exist. The conda-forge feedstock picks up the new release automatically.

## License

By contributing to cultivars, you agree that your contributions will be licensed under the project's [MIT License](https://github.com/nikhilxsunder/cultivars/blob/main/LICENSE). Every source file carries the MIT header (copyright 2026 Nikhil Sunder).

## Security Vulnerability Reporting

Please do **not** report security vulnerabilities through public GitHub issues. Instead, use GitHub Security Advisories ("Report a vulnerability" under the Security tab) or email nsunder724@gmail.com directly. Include a detailed description and reproduction steps, and allow time for a fix before public disclosure. We acknowledge receipt within 48 hours. See [`SECURITY.md`](SECURITY.md) for the full policy.

## Code of Conduct

This project adheres to our [Code of Conduct](https://github.com/nikhilxsunder/cultivars/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold it. Please report unacceptable behavior to nsunder724@gmail.com.

## Governance Model

cultivars follows a centralized governance model. The project owner and lead, Nikhil Sunder, has final authority on the project's direction, contributions, and dispute resolution. Contributors are encouraged to participate in discussions and submit pull requests; the project owner retains the right to approve or reject changes.

## Contact

For questions, open a GitHub Discussion or Issue, or reach out at nsunder724@gmail.com.