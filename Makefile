.PHONY: install-dev lint typecheck test smoke-eval train-dry-run package package-smoke verify benchmark benchmark-json studio compare-smoke lint-data

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
	"$$tmp_dir/bin/python" -c "import review_tuner; from review_tuner.studio import studio_root; assert review_tuner.__version__ == '0.9.0'; assert (studio_root() / 'index.html').is_file()"

verify: lint typecheck test smoke-eval lint-data compare-smoke package-smoke

smoke-eval:
	PYTHONPATH=src python -m review_tuner.evaluate \
		--golden data/golden/code_review_golden.jsonl \
		--predictions examples/predictions.sample.jsonl \
		--out reports/smoke_eval.json \
		--per-example-out reports/smoke_eval_examples.jsonl \
		--report-md reports/smoke_eval.md \
		--report-html reports/smoke_eval.html

lint-data:
	PYTHONPATH=src python -m review_tuner.lint \
		data/golden/code_review_golden.jsonl \
		data/train/code_review_comments.sample.jsonl

compare-smoke:
	PYTHONPATH=src python -m review_tuner.cli baseline \
		--golden data/golden/code_review_golden.jsonl \
		--out reports/baseline_predictions.jsonl
	PYTHONPATH=src python -m review_tuner.cli compare \
		--golden data/golden/code_review_golden.jsonl \
		--a reports/baseline_predictions.jsonl \
		--b examples/predictions.sample.jsonl \
		--out reports/compare_smoke.json

studio:
	PYTHONPATH=src python -m review_tuner.studio --port 8765 --open

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
