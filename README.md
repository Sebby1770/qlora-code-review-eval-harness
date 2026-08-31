# Review Tuner Studio

**Version 0.6.0**

Grade a code-review bot the way you would grade a teammate.

This repo still trains a compact causal LLM with QLoRA and scores it against a labelled golden set. What 0.4 adds is a **local studio** anyone can open: no GPU, no npm, no account. You click **Try the sample**, read a letter grade, and inspect the worst example — diff, expected comment, bot comment, and the phrases it skipped.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
review-eval studio --open
```

Then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). Diffs stay on your machine.

## Two audiences, one harness

| You want… | Use |
| --- | --- |
| To understand if a bot would catch a security hole | **Studio** (`review-eval studio`) |
| A CI gate on a golden JSONL | `review-eval eval … --fail-under 0.60` |
| To fine-tune Mistral 7B with QLoRA | `review-train-qlora` on a CUDA box |

Eval never downloads a model. Training is optional.

## What the studio shows you

- A **letter grade** (A–F) and a **Pass / Weak / Fail** against a threshold you can change
- Keyboard: **J** / **K** walk examples, **Esc** closes the inspector
- Copy the English summary or download the run as JSON
- One paragraph in English: what was skipped, how security cases did, and a 95% confidence band
- A **Start here** card for the worst example
- An inspector: unified diff, expected vs bot comment, required-phrase hits/misses, score waterfall
- Compare two prediction files (what newly caught / newly missed)
- A dataset linter so a messy JSONL fails before it grades

Advanced metrics (token F1, BLEU-lite, ROUGE-L) live behind a disclosure. You do not need them.

## Evaluation commands

```bash
# Score (baseline if --predictions omitted)
review-eval eval --golden data/golden/code_review_golden.jsonl \
  --report-md reports/eval.md \
  --report-html reports/eval.html \
  --report-json reports/eval.json

# Deterministic baseline predictions
review-eval baseline --golden data/golden/code_review_golden.jsonl --out predictions.jsonl

# Compare two prediction files
review-eval compare --golden data/golden/code_review_golden.jsonl pred_a.jsonl pred_b.jsonl

review-eval validate data/golden/code_review_golden.jsonl
review-eval lint data/golden/code_review_golden.jsonl
review-eval dashboard reports/
review-eval studio --port 8765
```

HTML reports now include a no-JavaScript **example inspector** (`<details>`) with the diff and comments.

Filters: `--language python --severity high --tag security`

## Dataset format

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

## How scoring works (short)

Exact string match is too brittle for review comments. The composite mixes:

| Weight | Signal | In English |
| ---: | --- | --- |
| 25% | token F1 | Shared words with the expected comment |
| 20% | required-phrase recall | The facts a good comment cannot skip |
| 15% | severity accuracy | Did it call a security hole “high”? |
| 10% | BLEU-lite | Short phrase overlap |
| 10% | ROUGE-L-lite | Longest matching word run |
| 10% | tag F1 | security / tests / performance, … |
| 5% | length ratio | Not wildly shorter or longer |
| 5% | 1 − forbidden rate | Did it avoid banned phrases? |

Required phrases are matched after lowercasing and stripping punctuation. A clever paraphrase that never uses those words will score poorly on that term — that is intentional and documented in the studio’s “How scoring works” page.

## Train with QLoRA (optional)

```bash
python -m pip install -e ".[train]"
huggingface-cli login   # if the base model is gated

PYTHONPATH=src python -m review_tuner.train_qlora \
  --config configs/qlora_mistral.yaml \
  --train data/train/code_review_comments.sample.jsonl \
  --eval data/golden/code_review_golden.jsonl \
  --dry-run
```

The default recipe is Mistral 7B Instruct, 4-bit NF4 LoRA. Adapter output: `outputs/code-review-mistral-qlora`.

Generate predictions, then score them:

```bash
PYTHONPATH=src python -m review_tuner.infer \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter outputs/code-review-mistral-qlora \
  --golden data/golden/code_review_golden.jsonl \
  --out reports/predictions.jsonl

review-eval eval --golden data/golden/code_review_golden.jsonl \
  --predictions reports/predictions.jsonl \
  --report-html reports/eval.html --fail-under 0.70
```

## Tests

```bash
make test
make smoke-eval
```

CI runs ruff, pytest, dataset lint, and a smoke eval on Python 3.11 and 3.12.

## Layout

```
src/review_tuner/           engine, CLI, studio server
src/review_tuner/studio_web  the local frontend (no build step)
data/golden/                 15 labelled review cases
data/train/                  tiny SFT sample
examples/                    sample predictions
```

## Metric references

- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [Transformers bitsandbytes docs](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
