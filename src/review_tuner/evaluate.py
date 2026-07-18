"""Command line evaluation harness for golden code review datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

from review_tuner.data import (
    index_by_id,
    iter_examples,
    iter_predictions,
    write_json,
    write_jsonl,
)
from review_tuner.metrics import (
    ExampleScore,
    ScoreAccumulator,
    ScoreComparison,
    ScoreComparisonAccumulator,
    aggregate_scores,
    score_example,
)
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def heuristic_prediction(example: ReviewExample) -> Prediction:
    """A deterministic baseline for smoke tests and harness validation."""

    diff_lower = example.diff.lower()
    evidence = f"{example.context}\n{example.file_path}\n{example.diff}".lower()
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

    if any(
        marker in evidence
        for marker in (
            "password",
            "token",
            "auth",
            "secret",
            "credential",
            "sql injection",
        )
    ):
        severity = "high"
    elif any(
        marker in evidence
        for marker in ("except", "catch", "select *", "for row in")
    ):
        severity = "medium"
    else:
        severity = "low"
    return Prediction(id=example.id, prediction=comment, severity=severity, tags=tags)


def evaluate_predictions(
    golden: Iterable[ReviewExample], predictions: Iterable[Prediction]
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    """Evaluate predictions against golden examples."""

    scores = list(iter_scores(golden, predictions))
    return aggregate_scores(scores), [score.as_dict() for score in scores]


def evaluate_comparisons(
    golden: Iterable[ReviewExample],
    candidate_predictions: Iterable[Prediction],
    baseline_predictions: Iterable[Prediction],
    *,
    top_regressions: int = 10,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate paired candidate and baseline predictions against one golden set."""

    accumulator = ScoreComparisonAccumulator(top_regressions=top_regressions)
    rows: list[dict[str, object]] = []
    for comparison in iter_score_comparisons(
        golden, candidate_predictions, baseline_predictions
    ):
        accumulator.add(comparison)
        rows.append(comparison.as_dict())
    return accumulator.report(), rows


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


def _labeled_prediction_index(
    predictions: Iterable[Prediction], label: str
) -> dict[str, Prediction]:
    try:
        return index_by_id(predictions)
    except DatasetError as exc:
        raise DatasetError(f"{label} predictions: {exc}") from exc


def iter_score_comparisons(
    golden: Iterable[ReviewExample],
    candidate_predictions: Iterable[Prediction],
    baseline_predictions: Iterable[Prediction],
) -> Iterator[ScoreComparison]:
    """Validate three-way alignment and yield paired scores in golden-set order."""

    candidate_by_id = _labeled_prediction_index(candidate_predictions, "candidate")
    baseline_by_id = _labeled_prediction_index(baseline_predictions, "baseline")
    missing_candidate: list[str] = []
    missing_baseline: list[str] = []
    golden_ids: set[str] = set()

    for example in golden:
        if example.id in golden_ids:
            raise DatasetError(f"duplicate golden id: {example.id}")
        golden_ids.add(example.id)
        candidate = candidate_by_id.get(example.id)
        baseline = baseline_by_id.get(example.id)
        if candidate is None:
            missing_candidate.append(example.id)
        if baseline is None:
            missing_baseline.append(example.id)
        if candidate is not None and baseline is not None:
            yield ScoreComparison(
                candidate=score_example(example, candidate),
                baseline=score_example(example, baseline),
            )

    if not golden_ids:
        raise DatasetError("golden dataset must contain at least one example")

    alignment_errors: list[str] = []
    for label, missing, indexed in (
        ("candidate", missing_candidate, candidate_by_id),
        ("baseline", missing_baseline, baseline_by_id),
    ):
        if missing:
            alignment_errors.append(
                f"missing {label} predictions for ids: {', '.join(missing)}"
            )
        extra = sorted(set(indexed) - golden_ids)
        if extra:
            alignment_errors.append(
                f"unexpected {label} predictions for ids: {', '.join(extra)}"
            )
    if alignment_errors:
        raise DatasetError("; ".join(alignment_errors))


def unit_interval(value: str) -> float:
    """Parse a command-line score threshold in the inclusive unit interval."""

    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative integer command-line value."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True, help="Path to golden JSONL examples.")
    parser.add_argument(
        "--predictions",
        help="Path to prediction JSONL. If omitted, a deterministic baseline is used.",
    )
    parser.add_argument(
        "--baseline-predictions",
        help="Known-good prediction JSONL to compare against the candidate predictions.",
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
        "--max-regression",
        type=unit_interval,
        default=None,
        help=(
            "Exit non-zero when candidate composite regresses from --baseline-predictions "
            "by more than this tolerance in [0, 1]."
        ),
    )
    parser.add_argument(
        "--top-regressions",
        type=non_negative_integer,
        default=10,
        help="Number of largest per-example composite regressions to include (default: 10).",
    )
    return parser


def _validate_cli_paths(args: argparse.Namespace) -> None:
    if args.predictions and args.write_baseline_predictions:
        raise DatasetError(
            "--write-baseline-predictions cannot be used with --predictions"
        )
    if args.max_regression is not None and not args.baseline_predictions:
        raise DatasetError("--max-regression requires --baseline-predictions")

    inputs = [
        ("--golden", args.golden),
        ("--predictions", args.predictions),
        ("--baseline-predictions", args.baseline_predictions),
    ]
    outputs = [
        ("--out", args.out),
        ("--per-example-out", args.per_example_out),
        ("--write-baseline-predictions", args.write_baseline_predictions),
    ]
    resolved_outputs: dict[Path, str] = {}
    for output_name, output_path in outputs:
        if output_path is None:
            continue
        resolved = Path(output_path).resolve()
        previous = resolved_outputs.get(resolved)
        if previous is not None:
            raise DatasetError(f"{output_name} and {previous} must be different paths")
        resolved_outputs[resolved] = output_name
        for input_name, input_path in inputs:
            if input_path is not None and resolved == Path(input_path).resolve():
                raise DatasetError(f"{output_name} must not overwrite {input_name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_cli_paths(args)
        if args.predictions:
            golden: Iterable[ReviewExample] = iter_examples(args.golden)
            predictions: Iterable[Prediction] = iter_predictions(args.predictions)
        else:
            golden = list(iter_examples(args.golden))
            predictions = [heuristic_prediction(example) for example in golden]

        if args.write_baseline_predictions:
            write_jsonl(
                args.write_baseline_predictions,
                (prediction.to_record() for prediction in predictions),
            )

        comparison_delta: float | None = None
        if args.baseline_predictions:
            comparison_accumulator = ScoreComparisonAccumulator(
                top_regressions=args.top_regressions
            )

            def comparison_rows() -> Iterator[dict[str, object]]:
                for comparison in iter_score_comparisons(
                    golden,
                    predictions,
                    iter_predictions(args.baseline_predictions),
                ):
                    comparison_accumulator.add(comparison)
                    yield comparison.as_dict()

            write_jsonl(args.per_example_out, comparison_rows())
            paired_report = comparison_accumulator.report()
            candidate_report = cast(
                dict[str, float | int], paired_report.pop("candidate")
            )
            delta_report = cast(dict[str, float | int], paired_report["delta"])
            comparison_delta = float(delta_report["composite"])
            report: dict[str, object] = {
                **candidate_report,
                "comparison": paired_report,
            }
        else:
            accumulator = ScoreAccumulator()

            def score_rows() -> Iterator[dict[str, float | str]]:
                for score in iter_scores(golden, predictions):
                    accumulator.add(score)
                    yield score.as_dict()

            write_jsonl(args.per_example_out, score_rows())
            candidate_report = accumulator.report()
            report = dict(candidate_report)

        write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))

        candidate_composite = float(candidate_report.get("composite", 0.0))
        gate_failures: list[str] = []
        if args.fail_under is not None and candidate_composite < args.fail_under:
            gate_failures.append(
                f"composite {candidate_composite:.4f} below threshold {args.fail_under:.4f}"
            )
        if (
            args.max_regression is not None
            and comparison_delta is not None
            and -comparison_delta > args.max_regression
            and not math.isclose(
                -comparison_delta,
                args.max_regression,
                abs_tol=1e-12,
            )
        ):
            gate_failures.append(
                f"composite regression {-comparison_delta:.4f} exceeds allowed "
                f"{args.max_regression:.4f}"
            )
        if gate_failures:
            print("; ".join(gate_failures), file=sys.stderr)
            return 2
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
