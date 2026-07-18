# QLoRA Code Review Comment Tuner

[![CI](https://github.com/Sebby1770/qlora-code-review-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Sebby1770/qlora-code-review-eval-harness/actions/workflows/ci.yml)

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
- Paired candidate-versus-baseline reports with regression gates and the worst regressed
  examples retained for diagnosis.
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

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make verify
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
  --out reports/predictions.jsonl \
  --batch-size 4 \
  --max-input-tokens 1024
```

Training and inference use the same structured target. Each generated JSONL row contains
`prediction`, `severity`, and `tags`, so every evaluator metric measures model output rather
than labels copied from the golden input. A shared formatter token-budgets only context and
diff: inference preserves the final JSON cue, while training reserves the complete structured
target and its stop token inside `training.max_length`. Training rejects an example when even
its instructions, metadata, complete target, and stop token cannot fit.

Adapters trained before this structured-output contract should be retrained. During a
transition, `review-infer --allow-unstructured-output` accepts a legacy plain-text comment,
but its unavailable severity and tag metadata will not receive credit. An explicit structured
`"tags": []` remains available metadata and receives credit when the golden tag set is empty.

Score the predictions:

```bash
PYTHONPATH=src python -m review_tuner.evaluate \
  --golden data/golden/code_review_golden.jsonl \
  --predictions reports/predictions.jsonl \
  --out reports/eval.json \
  --per-example-out reports/eval_examples.jsonl \
  --fail-under 0.70
```

Compare a candidate against checked-in or previously approved predictions and reject an
overall composite regression larger than one percentage point:

```bash
PYTHONPATH=src python -m review_tuner.evaluate \
  --golden data/golden/code_review_golden.jsonl \
  --predictions reports/candidate_predictions.jsonl \
  --baseline-predictions reports/approved_predictions.jsonl \
  --out reports/comparison.json \
  --per-example-out reports/comparison_examples.jsonl \
  --max-regression 0.01 \
  --top-regressions 20
```

The candidate metrics remain at the top level for compatibility. Comparison output is under
`comparison`, with baseline aggregates, candidate-minus-baseline deltas, win/tie/loss counts,
and a bounded `largest_regressions` list. Per-example rows keep the candidate fields at the
top level and add `baseline`, `delta`, and `outcome` fields. A positive delta is higher for
the candidate; for `forbidden_rate`, lower remains better.

For harness-only validation, omit `--predictions` and the evaluator will use the
deterministic baseline.

Measure synthetic scoring throughput with warmed, repeated runs:

```bash
make benchmark
```

Use `make benchmark-json` when collecting machine-readable results.

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

Training configuration is parsed into immutable typed sections before the ML stack is
loaded. Unknown options, invalid ranges, and incorrectly typed values fail fast with a
configuration error instead of reaching a long-running GPU job. The defaults in
`review_tuner.config` are also used by inference, keeping quantization policy consistent.

JSONL readers validate records lazily so training and inference do not retain the complete
source dataset in Python memory. List-shaped fields such as `tags` and rubric phrases must
be JSON arrays of non-empty strings. JSONL outputs are streamed through a temporary file
and atomically replace their destination only after generation succeeds.

Inference batches prompts while preserving record order, and the shared training/inference
formatter budgets the untrusted context and diff independently so truncation cannot remove
the task, output cue, or supervised target. Generated JSON is schema-validated before an
atomic prediction file replaces its destination. Evaluation requires an exact one-to-one ID
match: duplicate golden IDs,
missing predictions, unexpected stale predictions, and empty golden sets fail the run.
The model-agnostic generation core is tested independently of Torch and Transformers, while
model loading creates bitsandbytes configuration only when 4-bit loading is enabled.
Scoring normalizes each prediction and target once per example, then reuses those values
across exact-match, lexical, and rubric metrics. Severity and tag credit requires available
metadata rather than labels inferred from legacy comment prose. The deterministic smoke-test
baseline derives severity from prompt-visible context and diff signals, never from the golden
label. Composite metric weights are validated as one policy object before scoring.
Per-example scores stream directly to an atomic JSONL report while online statistics compute
aggregate means and population deviations in constant memory. Paired comparison uses the
same streaming path and retains only the configured number of worst regressions. Prompt
inputs are serialized as untrusted JSON and explicitly separated from the model instruction
hierarchy; gold severity and tags are targets, never prompt inputs.

Release-facing changes are recorded in [CHANGELOG.md](CHANGELOG.md).
See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request and report
vulnerabilities according to [SECURITY.md](SECURITY.md).
The verification workflow also installs the built wheel into an isolated environment and
executes its published CLI, catching distribution-only failures before release.

The QLoRA setup follows the Hugging Face PEFT quantization guide for 4-bit LoRA
training, TRL's `SFTTrainer` data path, and the Transformers bitsandbytes NF4
configuration guidance:

- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [Transformers bitsandbytes docs](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
