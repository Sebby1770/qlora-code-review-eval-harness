# Changelog

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
