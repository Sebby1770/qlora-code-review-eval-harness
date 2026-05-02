"""Command line evaluation harness for golden code review datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from review_tuner.data import index_by_id, load_examples, load_predictions, write_jsonl
from review_tuner.metrics import aggregate_scores, score_example
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def heuristic_prediction(example: ReviewExample) -> Prediction:
    """A deterministic baseline for smoke tests and harness validation."""

    diff_lower = example.diff.lower()
    if "password" in diff_lower or "token" in diff_lower or "auth" in diff_lower:
        comment = (
            "This looks security-sensitive. Please keep the existing validation path and add "
            "a regression test so expired or invalid tokens are rejected before state changes."
        )
        tags = ("security", "tests")
    elif "except" in diff_lower or "catch" in diff_lower:
        comment = (
            "This error handling hides the failure path. Please preserve the exception details "
            "or surface a clear typed error so callers can recover safely."
        )
        tags = ("reliability",)
    elif "select *" in diff_lower or "for row in" in diff_lower:
        comment = (
            "This may do unnecessary work as data grows. Please narrow the query or batch the "
            "loop and add a test that covers the large input case."
        )
        tags = ("performance", "tests")
    else:
        comment = (
            "Please add a focused regression test for this behavior and make the failure mode "
            "explicit for future maintainers."
        )
        tags = ("tests",)
    return Prediction(id=example.id, prediction=comment, severity=example.severity, tags=tags)


def evaluate_predictions(
    golden: list[ReviewExample], predictions: list[Prediction]
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    """Evaluate predictions against golden examples."""

    prediction_by_id = index_by_id(predictions)
    missing = [example.id for example in golden if example.id not in prediction_by_id]
    if missing:
        raise DatasetError(f"missing predictions for ids: {', '.join(missing)}")

    scores = [score_example(example, prediction_by_id[example.id]) for example in golden]
    return aggregate_scores(scores), [score.as_dict() for score in scores]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True, help="Path to golden JSONL examples.")
    parser.add_argument(
        "--predictions",
        help="Path to prediction JSONL. If omitted, a deterministic baseline is used.",
    )
    parser.add_argument("--out", default="reports/eval.json", help="Path for aggregate metrics.")
    parser.add_argument(
        "--per-example-out",
        default="reports/eval_examples.jsonl",
        help="Path for per-example scores.",
    )
    parser.add_argument(
        "--write-baseline-predictions",
        help="Optional path to write deterministic baseline predictions.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if composite score is below this value in [0, 1].",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        golden = load_examples(args.golden)
        if args.predictions:
            predictions = load_predictions(args.predictions)
        else:
            predictions = [heuristic_prediction(example) for example in golden]

        if args.write_baseline_predictions:
            write_jsonl(
                args.write_baseline_predictions,
                [
                    {
                        "id": prediction.id,
                        "prediction": prediction.prediction,
                        "severity": prediction.severity,
                        "tags": list(prediction.tags),
                    }
                    for prediction in predictions
                ],
            )

        aggregate, per_example = evaluate_predictions(golden, predictions)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_jsonl(args.per_example_out, per_example)
        print(json.dumps(aggregate, indent=2, sort_keys=True))

        if args.fail_under is not None and float(aggregate.get("composite", 0.0)) < args.fail_under:
            message = (
                f"composite {aggregate.get('composite', 0.0):.4f} "
                f"below threshold {args.fail_under:.4f}"
            )
            print(
                message,
                file=sys.stderr,
            )
            return 2
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
