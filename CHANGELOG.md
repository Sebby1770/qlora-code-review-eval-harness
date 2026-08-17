# Changelog

## [0.3.0] — 2026-08-18

### Added
- Error analysis: most-missed `must_mention` phrases, forbidden-phrase hits, severity confusion
- Deterministic bootstrap 95% CI on the aggregate composite score
- Self-contained HTML reports (`--report-html`)
- Optional JSON report copy (`--report-json`)
- `review-eval lint` dataset linter (duplicate ids, empty diff/comment, must_mention∩avoid, unknown severity)
- `review-eval dashboard` static index for a reports directory
- Stronger deterministic baseline that mentions deleted-line identifiers
- Six golden examples (`golden-010`–`golden-015`): SQL injection, N+1, missing authz, timezone/DST, resource leak, flaky test

### Changed
- Version **0.3.0**
- Markdown reports include CI, error analysis, and worst-example ranking
- Golden set expanded from 9 to 15 examples

## [0.2.0] — 2026-07-24

### Added
- Subcommands: `eval`, `compare`, `baseline`, `validate`
- Markdown reports (`--report-md`)
- Prediction comparison with metric deltas
- BLEU-lite and ROUGE-L-lite metrics, length ratio
- Aggregate breakdowns by language and severity
- Golden filters: `--language`, `--severity`, `--tag`
- Expanded golden set (9 examples)
- GitHub Actions CI (pytest, ruff, smoke eval)

### Changed
- Version **0.2.0**
- Composite score incorporates n-gram and length signals
