"""Command line evaluation harness for golden code review datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from review_tuner.data import (
    index_by_id,
    iter_examples,
    iter_predictions,
    write_json,
    write_jsonl,
)
from review_tuner.metrics import (
    ExampleScore,
    aggregate_scores,
    score_example,
)
from review_tuner.report import (
    DEFAULT_THRESHOLD,
    build_details,
    build_eval_report,
    render_html,
    render_markdown,
    write_text_report,
)
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def heuristic_prediction(example: ReviewExample) -> Prediction:
    """A deterministic baseline for smoke tests and harness validation."""

    diff_lower = example.diff.lower()
    tags: tuple[str, ...]
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
    golden: Iterable[ReviewExample], predictions: Iterable[Prediction]
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    """Evaluate predictions against golden examples."""

    scores = list(iter_scores(golden, predictions))
    return aggregate_scores(scores), [score.as_dict() for score in scores]


def iter_scores(
    golden: Iterable[ReviewExample], predictions: Iterable[Prediction]
) -> Iterator[ExampleScore]:
    """Validate exact dataset alignment while yielding scores in golden-set order."""

    prediction_by_id = index_by_id(predictions)
    missing: list[str] = []
    golden_ids: set[str] = set()
    for example in golden:
        if example.id in golden_ids:
            raise DatasetError(f"duplicate golden id: {example.id}")
        golden_ids.add(example.id)
        prediction = prediction_by_id.get(example.id)
        if prediction is None:
            missing.append(example.id)
            continue
        yield score_example(example, prediction)
    if not golden_ids:
        raise DatasetError("golden dataset must contain at least one example")

    extra = sorted(set(prediction_by_id) - golden_ids)
    alignment_errors = []
    if missing:
        alignment_errors.append(f"missing predictions for ids: {', '.join(missing)}")
    if extra:
        alignment_errors.append(f"unexpected predictions for ids: {', '.join(extra)}")
    if alignment_errors:
        raise DatasetError("; ".join(alignment_errors))


def unit_interval(value: str) -> float:
    """Parse a command-line score threshold in the inclusive unit interval."""

    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


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
        type=unit_interval,
        default=None,
        help="Exit non-zero if composite score is below this value in [0, 1].",
    )
    parser.add_argument(
        "--threshold",
        type=unit_interval,
        default=DEFAULT_THRESHOLD,
        help="Pass/weak/fail threshold used in reports. Default: 0.60.",
    )
    parser.add_argument("--report-md", help="Optional Markdown report path.")
    parser.add_argument("--report-html", help="Optional HTML report path.")
    return parser


def baseline_main(argv: list[str] | None = None) -> int:
    """Write heuristic baseline predictions for a golden JSONL."""

    parser = argparse.ArgumentParser(
        prog="review-eval baseline",
        description="Write deterministic heuristic predictions for a golden JSONL.",
    )
    parser.add_argument("--golden", required=True, help="Path to golden JSONL examples.")
    parser.add_argument("--out", required=True, help="Path to write prediction JSONL.")
    args = parser.parse_args(argv)
    try:
        examples = list(iter_examples(args.golden))
        if not examples:
            raise DatasetError("golden dataset must contain at least one example")
        predictions = [heuristic_prediction(example) for example in examples]
        write_jsonl(args.out, (prediction.to_record() for prediction in predictions))
        print(f"wrote {len(predictions)} baseline predictions to {args.out}")
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if Path(args.out).resolve() == Path(args.per_example_out).resolve():
            raise DatasetError("--out and --per-example-out must be different paths")
        if args.predictions and args.write_baseline_predictions:
            raise DatasetError(
                "--write-baseline-predictions cannot be used with --predictions"
            )
        examples = list(iter_examples(args.golden))
        if args.predictions:
            predictions = list(iter_predictions(args.predictions))
        else:
            predictions = [heuristic_prediction(example) for example in examples]

        if args.write_baseline_predictions:
            write_jsonl(
                args.write_baseline_predictions,
                (prediction.to_record() for prediction in predictions),
            )

        scores = list(iter_scores(examples, predictions))
        write_jsonl(args.per_example_out, (score.as_dict() for score in scores))
        aggregate = build_eval_report(
            examples,
            predictions,
            scores,
            threshold=args.threshold,
        )
        write_json(args.out, aggregate)
        details = build_details(examples, predictions, scores) if (
            args.report_md or args.report_html
        ) else []
        if args.report_md:
            write_text_report(args.report_md, render_markdown(aggregate, details))
        if args.report_html:
            write_text_report(args.report_html, render_html(aggregate, details))
        print(json.dumps(aggregate, indent=2, sort_keys=True))

        if (
            args.fail_under is not None
            and float(aggregate.get("composite", 0.0)) < args.fail_under
        ):
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
