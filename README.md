# QLoRA Code Review Eval Harness

[![CI](https://github.com/Sebby1770/qlora-code-review-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Sebby1770/qlora-code-review-eval-harness/actions/workflows/ci.yml)

Gate a review bot the way a staff engineer would: a golden set, a composite score you can fail CI on, HTML/Markdown reports, and a local studio that scores in the browser. QLoRA training is optional and never required for eval.

## Studio (start here)

The studio is static HTML/JS. It reimplements the Python scorer so you can drop JSONL files without installing Torch. Opening it loads the sample automatically, with filters, a score waterfall, and a severity confusion matrix.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make studio
```

Opens [http://127.0.0.1:8765/](http://127.0.0.1:8765/). Drop a golden JSONL and a predictions JSONL, or click **Try the sample**. `J` / `K` moves through examples. A second prediction file enables compare mode.

The same files are in `src/review_tuner/studio_web/` and can be published as GitHub Pages.

## CI gate

Eval has no ML dependencies. Composite weights are fixed:

`0.35` token F1 + `0.25` must-mention recall + `0.20` severity + `0.15` tag F1 + `0.05` forbidden-phrase absence.

```bash
review-eval --golden data/golden/code_review_golden.jsonl \
  --predictions examples/predictions.sample.jsonl \
  --out reports/eval.json \
  --per-example-out reports/eval_examples.jsonl \
  --report-md reports/eval.md \
  --report-html reports/eval.html \
  --fail-under 0.60
```

`review-eval --golden ...` still works (legacy flags). Subcommands:

| Command | Purpose |
| --- | --- |
| `review-eval eval` | Score predictions (same flags as above) |
| `review-eval baseline` | Write heuristic baseline predictions |
| `review-eval compare --golden G --a A.jsonl --b B.jsonl` | Newly caught / newly missed / Δ composite |
| `review-eval lint data/golden/code_review_golden.jsonl` | Duplicate ids, empty diffs, rubric issues |
| `review-eval slices --golden G --predictions P` | Mean composite by language, severity, tag |
| `review-eval studio` | Serve the local web studio |
| `review-eval report --eval eval.json --examples eval_examples.jsonl --report-html out.html` | Rebuild reports |

`review-eval` or `review-eval help` lists commands. `review-eval --help` shows the evaluator flags.

```bash
make verify          # lint, types, tests, smoke eval, data lint, compare smoke, wheel
make lint-data
make compare-smoke
```

Smoke eval writes `reports/smoke_eval.json`, `reports/smoke_eval_examples.jsonl`, `reports/smoke_eval.md`, and `reports/smoke_eval.html`.

## Dataset

Training rows use `review_comment`; golden rows use `expected_comment`.

```json
{
  "id": "golden-001",
  "language": "python",
  "file_path": "services/session.py",
  "context": "Session refresh should reject expired refresh tokens.",
  "diff": "@@ ...",
  "expected_comment": "This no longer rejects expired refresh tokens...",
  "severity": "high",
  "tags": ["security", "tests"],
  "rubric": {
    "must_mention": ["expired refresh tokens", "expires_at check"],
    "avoid": ["style"]
  }
}
```

`data/golden/code_review_golden.jsonl` has 12 examples across Python, TypeScript, Go, Rust, Java, and Ruby, covering security, performance, correctness, and tests. A couple of cases are written so the heuristic baseline misses them — a perfect A means the bot actually read the diff.

## Optional GPU train

Install the training extra only on a CUDA machine:

```bash
python -m pip install -e ".[train]"
huggingface-cli login   # if the base model is gated

PYTHONPATH=src python -m review_tuner.train_qlora \
  --config configs/qlora_mistral.yaml \
  --train data/train/code_review_comments.sample.jsonl \
  --eval data/golden/code_review_golden.jsonl \
  --dry-run
```

`review-train-qlora` and `review-infer` stay available. The default config is Mistral 7B NF4 QLoRA; swap `model.name` for a smaller local model when you want a cheaper run.

Generate predictions, then score them with `review-eval` as above:

```bash
PYTHONPATH=src python -m review_tuner.infer \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter outputs/code-review-mistral-qlora \
  --golden data/golden/code_review_golden.jsonl \
  --out reports/predictions.jsonl
```

## Metric notes

Exact string match is too brittle for review comments. The harness combines lexical overlap, rubric phrase recall, forbidden phrase rate, severity accuracy, and tag F1. Reports also include BLEU-lite, ROUGE-L-lite, length ratio, a security-fail bit (blocker/high with incomplete must-mention), language/severity/tag slices, a bootstrap 95% CI on composite (n≥3, seed 1337), a letter grade A–F, and pass/weak/fail against a 0.60 threshold.

See [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [SECURITY.md](SECURITY.md).
