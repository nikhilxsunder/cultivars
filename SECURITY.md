# Security Policy

## Table of Contents

- [Current Vulnerability Status](#current-vulnerability-status)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [Response Process](#response-process)
- [Disclosure Policy](#disclosure-policy)
- [Supported Versions](#supported-versions)
- [Threat Model](#threat-model)
- [Common Vulnerabilities and Mitigations](#common-vulnerabilities-and-mitigations)
  - [Compute-Library-Specific Vulnerabilities](#compute-library-specific-vulnerabilities)
  - [Optional Data-Loader Extra](#optional-data-loader-extra)
  - [General Python Vulnerabilities](#general-python-vulnerabilities)
- [Security Measures](#security-measures)
- [Third-Party Security Dependencies](#third-party-security-dependencies)
- [Secure Software Delivery](#secure-software-delivery)
  - [PyPI (Python Package Index)](#pypi-python-package-index)
  - [conda-forge](#conda-forge)
  - [GitHub Releases](#github-releases)
- [Verifying a Release](#verifying-a-release)
- [Security Updates and Announcements](#security-updates-and-announcements)
- [Security Design Principles](#security-design-principles)

## Current Vulnerability Status

cultivars is an early-stage (0.x) research-grade compute library. As of July 2026, there are no known unpatched vulnerabilities of medium or higher severity in the cultivars codebase.

We monitor for vulnerabilities through:

- Automated dependency scanning with GitHub Dependabot
- Static security analysis with CodeQL
- Community reports through our responsible disclosure process
- Regular manual security reviews

## Reporting a Vulnerability

The cultivars maintainer takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of:

- **GitHub Security Advisories**: use the "Report a vulnerability" button under the repository's _Security_ tab (preferred — this opens a private advisory)
- **Email**: nsunder724@gmail.com

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

## Response Process

When you report a vulnerability, you can expect:

1. **Acknowledgment**: We will acknowledge your report within 48 hours.
2. **Verification**: We will work to verify the vulnerability and its impact.
3. **Remediation**: We will develop a fix and test it.
4. **Disclosure**: Once a fix is ready, we will coordinate with you on the disclosure timeline.

## Disclosure Policy

- We will work with you to understand and address the vulnerability.
- We aim to provide a fix within 60 days of verification for all medium or higher severity vulnerabilities.
- We will credit you for the discovery in our release notes and `CHANGELOG` (unless you request otherwise).

## Supported Versions

cultivars is pre-1.0. Only the latest minor release receives security updates; there are no long-term support branches before 1.0, and the public API may change between minor releases (see `SemVer` policy in the architecture document).

| Version          | Supported |
| ---------------- | --------- |
| Latest 0.x minor | Yes       |
| Older 0.x minors | No        |

Once 1.0 ships, this table will be updated to track supported major lines.

## Threat Model

cultivars is a numerical/statistical **compute library**, not a network service or an API client. Its default operating mode reads in-memory arrays (`numpy`) and data frames (`pandas`), performs linear algebra, and returns typed result objects. This shapes the threat model:

- **The core performs no network I/O, opens no sockets, and requires no credentials.** Network access exists only in the optional data-loader extra (see below), and only when explicitly installed and used.
- **The primary attack surface is data, not traffic.** The two highest-consequence risks are (1) deserializing an untrusted serialized model/result object, which — like all `pickle`/`joblib` deserialization — can execute arbitrary code, and (2) resource exhaustion from adversarially large model specifications.
- **Inputs are validated at the `Spec` boundary.** An invalid specification cannot be constructed; validation (finite checks, shape/rank checks, bounded orders) happens before any computation.

## Common Vulnerabilities and Mitigations

The maintainer is familiar with the following vulnerability classes relevant to a scientific compute library and their mitigations.

### Compute-Library-Specific Vulnerabilities

1. **Insecure Deserialization of Model Artifacts**
   - **Vulnerability**: cultivars result objects are serializable (`io/serialize.py`, `joblib`-based). Loading a serialized artifact from an untrusted source can execute arbitrary code during unpickling — this is an inherent property of `pickle`/`joblib`, not a cultivars-specific bug.
   - **Mitigation**: Every serialized `Result` carries a `schema_version` that is validated on load. The documentation states unambiguously that serialized model files must be treated as executable code and loaded **only** from trusted sources. cultivars never auto-deserializes network input or files supplied by an untrusted party.

2. **Resource Exhaustion / Numerical Denial of Service**
   - **Vulnerability**: Pathological specifications (extreme lag orders, very large system dimensions, or enormous posterior draw counts) can exhaust memory or CPU.
   - **Mitigation**: `Spec` construction validates parameters against sane bounds and raises specific, descriptive exceptions rather than allocating unbounded work.

3. **Untrusted or Malformed Input Arrays**
   - **Vulnerability**: Non-finite values, wrong ranks, or non-conformable shapes propagating silently into linear algebra.
   - **Mitigation**: Defensive validation at every public boundary — finite checks, explicit shape/rank checks, and specific exception classes with comprehensive messages. No silent failure.

4. **Path Traversal in Serialization and Caching**
   - **Vulnerability**: Writing or reading cached/serialized data through unvalidated file paths.
   - **Mitigation**: Strict validation and normalization of file paths and names before any filesystem operation.

5. **Sensitive or Excessive Data in Logs and Errors**
   - **Vulnerability**: Leaking input data values or environment details through logs or exception messages.
   - **Mitigation**: Log messages and exceptions describe _shapes, types, and conditions_ rather than echoing data payloads; custom exception types filter what they surface.

### Optional Data-Loader Extra

cultivars ships an **optional** data-loader extra that bridges to external data sources (e.g. a FRED adapter via `fedfred`). This is the only component with network and credential surface, and the following mitigations apply **only when that extra is installed and used**:

1. **Insecure API Key Handling** — cultivars never stores API keys; keys are read from the environment or a secrets manager, and the documentation covers secure key management.
2. **Certificate Verification Bypass** — TLS certificate verification is always enforced in the HTTP client; it is never disabled.
3. **Injection / Unvalidated Parameters** — request parameters are validated before being sent upstream.
4. **Insecure Response Handling** — upstream responses are strictly type-validated before entering the compute path.

If you do not install the data-loader extra, none of the above surface exists.

### General Python Vulnerabilities

1. **Dependency-Chain Vulnerabilities**
   - **Vulnerability**: Security issues in dependencies (`numpy`, `scipy`, `pandas`, and any optional extras such as `jax`, `matplotlib`, or the data-loader stack).
   - **Mitigation**: Regular dependency scanning with Dependabot and a deliberately minimal core dependency footprint (`numpy` + `scipy` only in the substrate). Heavy dependencies are quarantined behind optional extras.

2. **Regular Expression Denial of Service (ReDoS)**
   - **Vulnerability**: Catastrophic backtracking in regular expressions used for parsing or validation.
   - **Mitigation**: cultivars uses regular expressions sparingly; those it does use are kept linear and simple by design.

3. **Improper Error Handling**
   - **Vulnerability**: Leaking sensitive information in error messages, or bare `except` clauses masking failures.
   - **Mitigation**: Custom exception types with controlled messaging; bare `except:` is banned and enforced by lint (`ruff E722`).

4. **Build and Release Supply Chain**
   - **Vulnerability**: Compromise of the publishing pipeline or leaked upload credentials.
   - **Mitigation**: Releases are published via PyPI Trusted Publishing (OIDC) with digital attestations — there are no long-lived upload tokens to leak (see [Secure Software Delivery](#secure-software-delivery)).

## Security Measures

cultivars implements several security and quality measures:

- Comprehensive type hints with `mypy --strict` on the public surface
- Static security analysis using CodeQL
- Dependency scanning and automated updates through GitHub Dependabot
- Defensive input validation with specific exception classes throughout
- Layered architecture with `import-linter`-enforced dependency boundaries (no backward imports; the substrate imports only `numpy` and `scipy`)
- Deterministic, explicitly-seeded randomness (module-level `np.random.seed` is banned and enforced by lint)
- Releases published via Trusted Publishing with digital attestations and signed git tags

## Third-Party Security Dependencies

cultivars relies on a small set of third-party libraries (`numpy` and `scipy` at the core; additional libraries behind optional extras). We track and update these dependencies to incorporate their security fixes.

## Secure Software Delivery

cultivars is distributed through channels that use TLS end to end, preventing interception and tampering in transit.

### PyPI (Python Package Index)

- All PyPI traffic uses HTTPS (TLS) by default.
- cultivars is published **exclusively via PyPI Trusted Publishing (OIDC)** from a pinned GitHub Actions workflow. No long-lived API tokens exist that could be leaked or misused.
- Every release carries [PEP 740](https://peps.python.org/pep-0740/) digital attestations establishing build provenance — which commit, which workflow, which repository produced the artifact — signed via Sigstore and displayed on the PyPI project page.
- Install securely using:
  ```sh
  pip install cultivars
  # or
  uv add cultivars
  ```

### conda-forge

- All conda-forge traffic uses HTTPS.
- Packages are built reproducibly by conda-forge CI from the public feedstock — never uploaded from a personal machine.
- Every artifact carries a SHA256 checksum.
- Install securely using:
  ```sh
  conda install -c conda-forge cultivars
  ```

### GitHub Releases

- All GitHub traffic uses HTTPS (TLS) by default.
- Release tags are cryptographically signed; GitHub displays a "Verified" badge on signed tags and commits.
- Clone securely using:
  ```sh
  git clone https://github.com/nikhilxsunder/cultivars.git
  ```

## Verifying a Release

cultivars releases can be verified two independent ways. Neither depends on downloading a detached `.asc` signature from the repository — PyPI no longer supports GPG/PGP release signatures, and build provenance is now established by attestations.

**1. Verify the signed git tag.**

```sh
git verify-tag v0.1.0
```

This confirms the tag was signed by the maintainer's key. The maintainer's public signing key is published on their GitHub profile (`https://github.com/nikhilxsunder.gpg` for GPG, or the account's SSH signing keys).

**2. Verify PyPI build provenance (attestations).**

Attestations are shown on the PyPI project page for each file. For artifacts built in GitHub Actions you can also verify provenance locally with the GitHub CLI:

```sh
gh attestation verify <downloaded-artifact> --repo nikhilxsunder/cultivars
```

This confirms the artifact was built by the expected repository and workflow, from a specific commit, rather than uploaded by hand.

## Security Updates and Announcements

Security updates will be announced via:

- GitHub Security Advisories
- GitHub release notes and the `CHANGELOG`
- The documentation portal

## Security Design Principles

cultivars follows established security design principles, adapted from Saltzer and Schroeder to the context of a scientific compute library:

1. **Economy of mechanism**: A single state-space substrate, a strictly layered architecture, and a `numpy` + `scipy`-only core keep the trusted computing base small.
2. **Fail-safe defaults**: Strict input validation, deterministic seeding, and — in the optional data loader — always-on TLS verification. No silent failure.
3. **Complete mediation**: Every input is validated at the `Spec` boundary before any computation; an invalid `Spec` cannot be constructed.
4. **Open design**: cultivars is MIT-licensed and fully open source. Security relies on correct design and verifiable provenance, never on obscurity; numerical correctness is validated by comparison against reference implementations.
5. **Separation of privilege**: The core requires no credentials at all. Where the optional data loader needs an API key, that key lives in the environment, separate from application code.
6. **Least privilege**: The core performs no network or filesystem I/O and holds no credentials. Capabilities that need more privilege (data loading, acceleration, plotting) are opt-in extras.
7. **Least common mechanism**: A layered design with no backward imports minimizes shared state; randomness is threaded through explicit generators rather than a global seed.
8. **Psychological acceptability**: A `statsmodels`-style, ergonomic API means the secure path is also the default, convenient path.
9. **Limited attack surface**: A minimal core, optional heavy dependencies behind extras, and `import-linter`-enforced boundaries keep exposure small.
10. **Input validation with allowlists**: All parameters are validated against explicit constraints — finite checks, shape and rank checks, and bounded model orders — before processing.

---

For questions about this policy, please contact nsunder724@gmail.com.
