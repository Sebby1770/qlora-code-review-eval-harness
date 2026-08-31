# Review Tuner 0.5 — Studio Plan

**From:** 0.4.1  
**To:** 0.5.0 — guide for first-timers, grade histogram, tag breakdown, live gate, theme, HTML export, paste check, run history

Eval stays stdlib. Frontend stays vanilla JS.

## Why

0.4 lets anyone grade a sample. 0.5 makes the desk usable for a second visit: you can restamp the pass gate, see the score distribution, export a shareable HTML report, and not get lost on paste-JSONL.

## Work packages

1. **Guide** `/#/guide` — four plain-language steps with a button that runs the sample.
2. **Histogram + by_tag** on the eval view (letter counts; mean composite per golden tag).
3. **Live threshold slider** on results — restamp Pass/Weak/Fail without re-scoring.
4. **Light theme** toggle, persisted in localStorage.
5. **Paste lint** — count JSONL rows or show a parse error before Grade.
6. **Overlap tokens** on each example (shared words gold vs bot) shown in the inspector.
7. **Download HTML report** via `POST /api/report-html` using the existing renderer.
8. **Run history** — last five letter/story summaries on Home (sessionStorage).
9. Tests for histogram, by_tag, overlap, report-html, sample tokens still work.
10. VERSION 0.5.0.

---

# Review Tuner 0.4 — Studio Plan

**From:** review-tuner 0.3 CLI harness (JSON/MD/HTML reports, no application)  
**To:** 0.4.0 — a local studio anyone can open, plus a richer inspector in static HTML reports

The eval engine stays **stdlib-only**. GPU training remains optional. The studio is a stdlib HTTP server serving a designed frontend. No npm, no React, no CDN required at runtime (fonts may use a system stack so the app works offline).

---

## Who it is for

| Person | First minute | They never have to see |
| --- | --- | --- |
| Staff engineer / EM | Click **Try the sample**, read Pass/Weak/Fail | LoRA, BLEU, NF4, JSONL |
| Reviewer writing goldens | Open Dataset, paste a diff, add “must mention” | Composite weights |
| ML engineer | Advanced panel, thresholds, compare two JSONL files | Nothing — power is there |
| New hire | Guided 3-step grade | The CLI |

Primary language in the UI: **grade + plain English**. Composite scores live one click deeper.

---

## Product name and metaphor

**Review Tuner Studio** — a senior engineer’s review desk.

Not a purple “AI dashboard”. Visual system:

- Ink `#0e1a17` background, paper `#f3efe4` cards, forest `#1f6f5b`, amber `#e3b23c` for the grade
- IBM Plex-like stack via system fonts: `"IBM Plex Sans", "iA Writer Quattro", "Segoe UI", system-ui`
- Mono for diffs: `"IBM Plex Mono", ui-monospace, Menlo, monospace`
- Large editorial grade letter (A / B / C / D / F) with a one-sentence story
- Motion: short fades, no particle soup

---

## Information architecture

Single-page app with hash routes, all in `studio_web/`:

1. **Home** `/#/` — promise, three doors
   - Try the bundled sample (golden + sample predictions)
   - Grade with the built-in baseline (no model)
   - Upload my JSONL files
2. **Guide** `/#/guide` — 4 steps with copy a non-ML person can follow
3. **Grade** `/#/grade` — run eval, live progress, cancel
4. **Results** `/#/run` — the money screen
5. **Inspector** — example drawer on the results page (diff, gold, model, phrases)
6. **Compare** `/#/compare` — two prediction files, flip list
7. **Dataset** `/#/dataset` — visual linter + coverage map
8. **How scoring works** `/#/how` — glossary in human words, then the formula

Persistent chrome: wordmark, grade/compare/dataset/how, “Advanced” toggle, theme.

---

## Results screen (must-have)

Hero:

- Letter grade from composite: A ≥ 0.80, B ≥ 0.65, C ≥ 0.50, D ≥ 0.35, else F
- Label: Pass / Weak / Fail vs a default threshold 0.60 (editable)
- One sentence, generated from error analysis, e.g.  
  “The bot missed authorization language on 3 of 4 security cases, and overall scored 54%.”
- CI in English: “We are 95% sure the true score sits between 46% and 63%.”

Body:

- **Start here** card: worst example, pinned
- Failure story: most-missed phrases as chips
- Language × severity heatmap (CSS grid)
- Example table: filter language / severity / tag / failures only; click opens inspector
- Advanced accordion: token F1, BLEU-lite, ROUGE-L, weights

Inspector:

- Unified diff, syntax-neutral but colored +/− lines
- Gold comment vs model comment
- Must-mention phrases highlighted as hit (green) or miss (red)
- Forbidden hits highlighted
- Score waterfall: each composite term as a bar with a one-line why

---

## Backend

New module `review_tuner/studio.py` + `review_tuner/studio_web/`.

```
review-eval studio --host 127.0.0.1 --port 8765 --open
```

Stdlib `http.server`. JSON APIs:

| Method | Path | Role |
| --- | --- | --- |
| GET | `/` and `/assets/*` | Static studio |
| GET | `/api/health` | version, stdlib flag |
| GET | `/api/glossary` | human metric copy |
| GET | `/api/samples/golden` | bundled golden JSONL as JSON array |
| GET | `/api/samples/predictions` | bundled sample preds |
| GET | `/api/samples/baseline` | baseline preds for bundled golden |
| POST | `/api/eval` | `{golden, predictions?, filters?}` → aggregate + examples **with text** |
| POST | `/api/lint` | `{records}` or `{text}` |
| POST | `/api/compare` | `{golden, predictions_a, predictions_b}` |
| POST | `/api/baseline` | `{golden}` |

CORS not needed (same origin). Body size cap ~8 MB. Bind default **127.0.0.1**.

### Eval payload (the UI contract)

Each example in the response must include:

```
id, language, file_path, context, diff, expected_comment, prediction,
severity, predicted_severity, tags, predicted_tags,
must_mention, avoid, missed_must_mention, forbidden_hits,
metrics..., composite, grade, explanation[]
```

`explanation` is a list of `{key, weight, value, contribution, plain}` for the waterfall.

Also return `story` (plain-English paragraph) and `letter_grade`.

Implement `build_eval_view(...)` in Python so HTML reports and the studio share one shape.

---

## Engine hardening (small, required for honest UI)

- Ordered severity matching in `infer_severity` (`blocker, high, medium, low, nit`)
- `ExampleScore.as_dict` may grow fields; keep old keys
- Per-example output from CLI `--per-example-out` optionally includes text when `--include-text` is set (studio always includes text)
- Extra predictions: warn, do not silently drop from the UI (list unmatched ids)

---

## Static HTML report upgrade

`render_html_report` gains an example inspector (details/summary, no JS required). Worst example first. Diff + comments + phrase chips. Keep self-contained inline CSS, no CDN. This is what CI still publishes.

---

## Files

```
src/review_tuner/studio.py
src/review_tuner/studio_web/index.html
src/review_tuner/studio_web/styles.css
src/review_tuner/studio_web/app.js
src/review_tuner/view.py          # shared eval view / story / grade
tests/test_studio.py
tests/test_view.py
```

Package data: `studio_web/*`. Script: `review-eval studio`.

---

## Tests

- `letter_grade` boundaries
- `story` mentions a missed phrase when error analysis is non-empty
- POST `/api/eval` with in-memory golden+preds returns 200 and example diffs
- Sample eval matches existing smoke composite ≈ 0.5438
- Lint API returns errors for duplicate ids
- Compare API delta signs
- HTML report contains a `<details>` inspector
- `review-eval studio --help`
- Existing pytest suite still green; ruff clean

---

## Out of scope

GPU train console, embedding model, LLM-as-judge, accounts, deploying the studio to the public internet (it is local-first).

---

## Acceptance

```
pip install -e ".[dev]"
pytest -q
ruff check src tests
review-eval studio --port 8765
```

Opening the page: click **Try the sample**, see a letter grade, open the worst example, understand *why* it failed without reading this plan.
