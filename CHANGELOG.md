# Changelog

All notable changes to this project will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-08

### Changed

- Studio frontend is now the Cyanotype print lab: Prussian-blue sun-print stage,
  circular exposure meter, glass-plate drop zones, and contact-sheet inspector.

## [0.9.0] - 2026-09-07

### Added

- Studio auto-loads the sample run, with language/severity filters, a per-example
  score waterfall, severity confusion matrix, copy-summary, and download-JSON.
- In-browser bootstrap 95% CI (mulberry32, seed 1337) so GitHub Pages shows a band
  without Python.

## [0.8.0] - 2026-09-07

### Added

- Unified `review-eval` CLI with `eval`, `baseline`, `compare`, `lint`, `slices`,
  `studio`, and `report` subcommands. Legacy `review-eval --golden ...` still works.
- Local static studio (`review-eval studio`) with an in-browser scorer that matches
  the Python composite weights, sample JSONL, J/K inspector, and optional compare.
- Additive metrics: BLEU-lite, ROUGE-L-lite, length ratio, and security-fail.
- Aggregate extras: language/severity/tag slices, bootstrap 95% CI (seed 1337),
  letter grade A–F, and pass/weak/fail against a 0.60 threshold.
- Markdown and HTML evaluation reports (`--report-md`, `--report-html`).
- Dataset linter for duplicate ids, empty diffs, missing rubrics, short comments,
  and overlapping must_mention/avoid phrases.
- Prediction comparer that prints newly caught / newly missed / Δ composite.
- Expanded golden set to 12 examples (Python, TypeScript, Go, Rust, Java, Ruby)
  including cases the heuristic baseline should miss.
- Makefile targets `studio`, `lint-data`, and `compare-smoke`.

### Changed

- Package version is `0.8.0`.
- `review-eval` console script now points at `review_tuner.cli:main`.
- README leads with the studio and CI gate; GPU training is optional.
- Smoke eval also writes Markdown and HTML into `reports/`.

### Fixed

- Public `ExampleScore.as_dict` keys and `heuristic_prediction` behavior are
  unchanged; extra metrics sit beside that contract.

## [0.1.0]

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

[Unreleased]: https://github.com/Sebby1770/qlora-code-review-eval-harness/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Sebby1770/qlora-code-review-eval-harness/compare/v0.1.0...v0.8.0
[0.1.0]: https://github.com/Sebby1770/qlora-code-review-eval-harness/releases/tag/v0.1.0
