.PHONY: install-dev lint typecheck test smoke-eval train-dry-run package package-smoke verify benchmark benchmark-json

install-dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests benchmarks

typecheck:
	mypy src/review_tuner

test:
	pytest -q

package:
	python -m build
	twine check dist/*

package-smoke: package
	@tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	python -m venv "$$tmp_dir"; \
	"$$tmp_dir/bin/python" -m pip install --quiet --no-deps dist/*.whl; \
	"$$tmp_dir/bin/review-eval" --help >/dev/null; \
	"$$tmp_dir/bin/python" -c "import review_tuner; assert review_tuner.__version__ == '0.1.0'"

verify: lint typecheck test smoke-eval package-smoke

smoke-eval:
	PYTHONPATH=src python -m review_tuner.evaluate \
		--golden data/golden/code_review_golden.jsonl \
		--predictions examples/predictions.sample.jsonl \
		--out reports/smoke_eval.json \
		--per-example-out reports/smoke_eval_examples.jsonl

train-dry-run:
	PYTHONPATH=src python -m review_tuner.train_qlora \
		--config configs/qlora_mistral.yaml \
		--train data/train/code_review_comments.sample.jsonl \
		--eval data/golden/code_review_golden.jsonl \
		--dry-run

benchmark:
	PYTHONPATH=src python benchmarks/bench_evaluation.py

benchmark-json:
	PYTHONPATH=src python benchmarks/bench_evaluation.py --json
