# Changelog

## [0.6.1] — 2026-08-31

### Changed
- README clone URL and version so GitHub `main` is the studio product.

## [0.6.0] — 2026-08-31

### Added
- SVG grade badge (`GET /api/badge`, `POST /api/badge`) and Download badge
- Histogram bars, sortable example table, id search, heatmap click-to-filter
- Severity confusion table on the results page
- `?` keyboard cheat sheet
- Dataset “Insert a starter row”
- Last eval payload remembered for HTML/badge download in the same session

### Changed
- Version **0.6.0**

## [0.5.0] — 2026-08-31

### Added
- Guide page (`/#/guide`) for first-time users
- Letter-grade histogram and mean composite by golden tag
- Live pass-gate slider on the results page (no re-score)
- Light “paper” theme, persisted in localStorage
- Paste JSONL row counter / parse error before grading
- Shared-word list in the example inspector
- `POST /api/report-html` plus Download HTML report
- Last-five run history on Home (session only)

### Changed
- Version **0.5.0**

## [0.4.1] — 2026-08-31

### Added
- Studio shortcuts: POST `"golden": "sample"` and `"predictions": "sample"|"baseline"`
- Dataset coverage map (language / severity / tag) on the lint API and Dataset page
- Inspector phrase highlighting, previous/next, and J/K/Esc keyboard navigation
- Copy English summary, download run JSON, toasts instead of `alert`
- Compare: click a flipped example to read A vs B comments
- Tag filter on the results table

### Changed
- Version **0.4.1**
- Home doors and forms talk to the sample/baseline tokens so the UI does less file shuffling

## [0.4.0] — 2026-08-31

### Added
- **Review Tuner Studio** — local stdlib HTTP app (`review-eval studio`) with a designed frontend
- Guided sample / baseline / upload grading, example inspector, heatmap, compare, and dataset linter
- Shared eval view model: letter grade, Pass/Weak/Fail, English story, score waterfall
- HTML reports include a no-JS example inspector (diff, expected comment, bot comment, phrase chips)
- `--include-text` on `review-eval eval` for diffs/comments in the per-example JSONL
- Bundled studio samples so the UI works without extra files

### Changed
- Version **0.4.0**
- Severity inference uses a stable label order (`blocker` → `nit`) instead of set iteration
- Aggregate JSON reports include `letter_grade` and `story`

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
