# Contributing

Thanks for improving the review-tuning harness.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make verify
```

Install `.[train]` only when working on GPU training or model inference. Keep lightweight
evaluation usable without ML dependencies.

## Project rules

- Preserve the typed configuration and validated JSONL boundaries.
- Keep training and inference prompt construction on the shared prompt path.
- Add focused tests for schema, metric, CLI, or report-contract changes.
- Keep total test coverage at or above the enforced 75% floor.
- Do not commit model weights, tokens, generated reports, or local environment files.
- Record user-visible changes under `Unreleased` in `CHANGELOG.md`.

Pull requests should explain the behavior change, verification performed, and any metric or
dataset compatibility impact.
