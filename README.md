# QLoRA Code Review Comment Tuner

**Version 0.2.0**


Fine-tune a compact causal LLM to write code review comments, then score it with a
repeatable golden-set harness. The repo is intentionally split into two paths:

- `review-train-qlora`: GPU training with Transformers, PEFT, TRL, and bitsandbytes.
- `review-eval`: lightweight evaluation that runs locally without downloading a model.

The default config uses Mistral 7B with 4-bit NF4 QLoRA. You can swap
`model.name` for a smaller local or Hugging Face model when you want a cheaper run.

## What is included

- QLoRA training script for supervised fine-tuning on JSONL code review examples.
- Prompt formatter that turns diffs, context, severity, and tags into SFT rows.
- Golden dataset format with rubric phrases, forbidden phrases, severity, and tags.
- Automated evaluation harness with lexical F1, rubric phrase recall, forbidden phrase
  rate, severity accuracy, tag F1, and a composite score.
- Deterministic baseline predictions for smoke tests.
- CI workflow that lints, tests, and runs a smoke evaluation.

## Dataset Format

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


## Evaluation commands (0.2)

```bash
# Score (baseline if --predictions omitted)
review-eval eval --golden data/golden/code_review_golden.jsonl --report-md reports/eval.md

# Deterministic baseline predictions
review-eval baseline --golden data/golden/code_review_golden.jsonl --out predictions.jsonl

# Compare two prediction files
review-eval compare --golden data/golden/code_review_golden.jsonl pred_a.jsonl pred_b.jsonl

# Validate dataset schema
review-eval validate data/golden/code_review_golden.jsonl
```

Filters: `--language python --severity high --tag security`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
make smoke-eval
```

The smoke evaluation writes:

- `reports/smoke_eval.json`
- `reports/smoke_eval_examples.jsonl`

## Train With QLoRA

Install the optional training stack on a CUDA-capable machine:

```bash
python -m pip install -e ".[train]"
```

If the base model is gated, authenticate with Hugging Face first:

```bash
huggingface-cli login
```

Run a config/data validation pass:

```bash
PYTHONPATH=src python -m review_tuner.train_qlora \
  --config configs/qlora_mistral.yaml \
  --train data/train/code_review_comments.sample.jsonl \
  --eval data/golden/code_review_golden.jsonl \
  --dry-run
```

Launch training:

```bash
PYTHONPATH=src python -m review_tuner.train_qlora \
  --config configs/qlora_mistral.yaml \
  --train data/train/code_review_comments.sample.jsonl \
  --eval data/golden/code_review_golden.jsonl
```

The LoRA adapter is saved to `outputs/code-review-mistral-qlora`.

## Generate And Evaluate

Generate model predictions:

```bash
PYTHONPATH=src python -m review_tuner.infer \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter outputs/code-review-mistral-qlora \
  --golden data/golden/code_review_golden.jsonl \
  --out reports/predictions.jsonl
```

Score the predictions:

```bash
PYTHONPATH=src python -m review_tuner.evaluate \
  --golden data/golden/code_review_golden.jsonl \
  --predictions reports/predictions.jsonl \
  --out reports/eval.json \
  --per-example-out reports/eval_examples.jsonl \
  --fail-under 0.70
```

For harness-only validation, omit `--predictions` and the evaluator will use the
deterministic baseline.

## Metric Philosophy

Exact string match is too brittle for review comments, so the harness combines:

- lexical overlap against the reference comment,
- rubric phrase recall for issue-specific must-haves,
- forbidden phrase rate to catch unwanted style-only comments,
- severity classification accuracy,
- tag F1 for issue category alignment.

This gives you a stable gate for regression testing while still allowing natural
language variation.

## Implementation Notes

The QLoRA setup follows the Hugging Face PEFT quantization guide for 4-bit LoRA
training, TRL's `SFTTrainer` data path, and the Transformers bitsandbytes NF4
configuration guidance:

- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [Transformers bitsandbytes docs](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
