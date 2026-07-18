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
- Structured JSON targets shared by training and inference, with strict generated-output
  validation and an explicit legacy plain-text fallback.
- Paired candidate-versus-baseline evaluation, configurable regression gates, outcome counts,
  and bounded worst-regression diagnostics.

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
- Training and inference now share field-aware token budgeting: inference preserves the
  structured-output cue, and training reserves the complete supervised target.
- Golden severity and tag labels are emitted only as supervised targets, not included in
  model prompt inputs.
- The deterministic smoke-test baseline derives severity from prompt-visible signals instead
  of copying the golden label.

### Fixed

- Prevented malformed strings and nulls from silently becoming invalid tag/rubric sequences.
- Prevented stale or duplicate records from producing credible-looking evaluation reports.
- Prevented failed generation or alignment checks from replacing existing reports.
- Prevented baseline-output options from consuming supplied prediction iterators.
- Prevented non-4-bit training from constructing bitsandbytes configuration or running
  k-bit-only model preparation.
- Prevented multi-label severity prose from resolving nondeterministically.
- Prevented generation from reaching Transformers without a usable padding token id.
- Prevented evaluation reports from overwriting golden or prediction inputs through colliding
  CLI paths.
- Prevented malformed or incomplete model output, including null severity and non-array tags,
  from silently losing severity and tag scores.
- Prevented legacy predictions with unavailable metadata from earning severity or tag credit;
  explicit structured empty tag arrays remain distinguishable from missing tags.
- Prevented `training.max_length` from silently truncating the output cue, structured target,
  or trainer-added EOS token.
- Raised the TRL dependency floor to the first release that supports the configured
  `SFTConfig.max_length` and `SFTTrainer.processing_class` APIs.
- Preserved actionable schema and token-budget errors across Arrow dataset generation
  wrappers instead of exposing an opaque training traceback.

### Security

- Added strict configuration and dataset validation before expensive model work begins.
- Isolated untrusted diff/context content from model instructions.

[Unreleased]: https://github.com/Sebby1770/qlora-code-review-eval-harness/compare/v0.1.0...HEAD
