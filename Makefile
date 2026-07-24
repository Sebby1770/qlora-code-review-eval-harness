.PHONY: install-dev lint test smoke-eval train-dry-run

install-dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest -q

smoke-eval:
	PYTHONPATH=src python -m review_tuner.evaluate eval \
	--golden data/golden/code_review_golden.jsonl \
	--predictions examples/predictions.sample.jsonl \
	--out reports/smoke_eval.json \
	--per-example-out reports/smoke_eval_examples.jsonl \
	--report-md reports/smoke_eval.md

train-dry-run:
	PYTHONPATH=src python -m review_tuner.train_qlora \
		--config configs/qlora_mistral.yaml \
		--train data/train/code_review_comments.sample.jsonl \
		--eval data/golden/code_review_golden.jsonl \
		--dry-run
