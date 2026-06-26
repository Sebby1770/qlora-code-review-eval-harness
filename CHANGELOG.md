# Changelog

All notable changes to this project will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Immutable, validated configuration models shared by training and inference.
- Lazy JSONL readers, atomic JSON/JSONL writers, and typed package metadata.
- Batched inference with explicit input and output token budgets.
- Constant-memory online aggregation for evaluation metrics.
- Validated composite score weights and a reusable prediction serialization helper.
- Strict typing, expanded schema/evaluation tests, and GitHub Actions verification.
- `@Sebby1770` code ownership and grouped Dependabot updates.
- Contribution, security, pull-request, and structured bug-report guidance.
- Coverage and package-build gates for release verification.
- A model-agnostic inference core with fake-runtime tests for batching and output ordering.
- A repeatable synthetic evaluation-throughput benchmark.
- Isolated built-wheel installation and published-entry-point smoke verification.

### Changed

- Evaluation now requires an exact one-to-one match between golden and prediction IDs.
- Training datasets stream into Arrow-backed datasets instead of first becoming Python lists.
- Review context and diffs are serialized as explicitly untrusted JSON prompt data.
- Quantization defaults and optional-dependency errors have one source of truth.
- Severity labels now have one canonical ordering shared by schema validation and metrics.
- Package version metadata now derives from the runtime `__version__` source.
- Licensing metadata uses the current SPDX format supported by setuptools.
- The enforced test coverage floor increased from 65% to 75%.
- Per-example scoring now reuses normalized text and avoids reflective metric aggregation
  in the hot path.

### Fixed

- Prevented malformed strings and nulls from silently becoming invalid tag/rubric sequences.
- Prevented stale or duplicate records from producing credible-looking evaluation reports.
- Prevented failed generation or alignment checks from replacing existing reports.
- Prevented baseline-output options from consuming supplied prediction iterators.
- Prevented non-4-bit training from constructing bitsandbytes configuration or running
  k-bit-only model preparation.
- Prevented multi-label severity prose from resolving nondeterministically.
- Prevented generation from reaching Transformers without a usable padding token id.

### Security

- Added strict configuration and dataset validation before expensive model work begins.
- Isolated untrusted diff/context content from model instructions.

[Unreleased]: https://github.com/Sebby1770/qlora-code-review-eval-harness/compare/v0.1.0...HEAD
