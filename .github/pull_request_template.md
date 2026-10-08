## Summary

<!-- Describe what this PR does and why. Focus on the user-visible behavior in cultivars. -->

## Related Issues / Discussions

<!-- e.g. Closes #123, Related to #456 -->

## Type of Change

- [ ] Bug fix
- [ ] New feature (new model, identification scheme, prior, or estimator)
- [ ] Numerical correctness fix / reference-comparison update
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactor / internal change
- [ ] CI / tooling / packaging

## Public API Impact

- [ ] No public API changes
- [ ] Adds new public API (functions, classes, arguments, or return types)
- [ ] Changes existing public API
- [ ] Removes or deprecates public API
- [ ] **Changes a locked decision in `architecture.md` §6** (requires a major version bump and a doc update in the same PR)

If there **are** API changes, describe them clearly (including any breaking changes):

<!-- e.g. "VAR(y=df, p=4).fit() now returns a VARResult with a .diagnostics sub-object instead of exposing residuals at the top level" -->

## Architecture Compliance

<!-- Confirm the change respects the binding architecture. Tick what applies. -->

- [ ] Respects layer boundaries — no upward/backward imports (`import-linter` passes)
- [ ] Follows the three-object discipline (one `Spec`, one `Estimator`, one `Result` per file)
- [ ] Composition via strategy objects, not new subclasses (no `BVAR_Minnesota_SV_*` classes)
- [ ] Substrate purity preserved (`core/` and `state_space/` import only `numpy` + `scipy`)
- [ ] MIT header present on any new `.py` file
- [ ] N/A — this PR does not touch library code

## How to Test

<!-- Provide concrete steps / commands for reviewers. -->

```bash
# Full suite
uv run pytest

# Targeted, e.g. a specific module or property tests
uv run pytest tests/core -k stability
uv run pytest -k property --hypothesis-show-statistics

# Coverage
uv run pytest --cov=cultivars --cov-report=term-missing

# Docs (if relevant)
cd docs && make html
```

## Checklist

- [ ] Tests added/updated and pass locally
- [ ] Coverage ≥ 90% overall (100% branch on `core/` if touched)
- [ ] Reference-comparison test added/updated if this adds or changes an estimator (vs. statsmodels / R / analytic values)
- [ ] `uv run mypy src/cultivars/`, `uv run ruff check`, `uv run ruff format --check`, and `uv run import-linter` all pass
- [ ] Docs updated (Google-style docstrings and/or Sphinx) if behavior or API changed
- [ ] Examples and `CHANGELOG` updated if relevant
- [ ] Commits are signed off (DCO: `git commit -s`)
- [ ] No unexpected changes to generated files (docs build artifacts, cache, lockfile churn, etc.)

## Additional Context

<!-- Anything reviewers should know: references implemented (with citation), known limitations, follow-up work. -->
