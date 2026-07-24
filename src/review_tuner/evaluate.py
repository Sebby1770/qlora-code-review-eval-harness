"""Command line evaluation harness for golden code review datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from review_tuner import __version__
from review_tuner.data import index_by_id, load_examples, load_predictions, write_jsonl
from review_tuner.metrics import aggregate_scores, render_markdown_report, score_example
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


def filter_examples(
    golden: list[ReviewExample],
    *,
    language: str | None = None,
    severity: str | None = None,
    tag: str | None = None,
) -> list[ReviewExample]:
    """Filter golden examples by optional language, severity, or tag."""

    out = golden
    if language:
        out = [ex for ex in out if ex.language.lower() == language.lower()]
    if severity:
        out = [ex for ex in out if ex.severity.lower() == severity.lower()]
    if tag:
        needle = tag.lower()
        out = [ex for ex in out if needle in ex.tags]
    return out


def evaluate_predictions(
    golden: list[ReviewExample], predictions: list[Prediction]
) -> tuple[dict, list[dict]]:
    """Evaluate predictions against golden examples."""

    prediction_by_id = index_by_id(predictions)
    missing = [example.id for example in golden if example.id not in prediction_by_id]
    if missing:
        raise DatasetError(f"missing predictions for ids: {', '.join(missing)}")

    scores = [score_example(example, prediction_by_id[example.id]) for example in golden]
    return aggregate_scores(scores), [score.as_dict() for score in scores]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_eval(args: argparse.Namespace) -> int:
    golden = filter_examples(
        load_examples(args.golden),
        language=args.language,
        severity=args.severity,
        tag=args.tag,
    )
    if not golden:
        raise DatasetError("no golden examples remain after filters")

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
    _write_json(out_path, aggregate)
    write_jsonl(args.per_example_out, per_example)
    if args.report_md:
        md_path = Path(args.report_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown_report(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))

    if args.fail_under is not None and float(aggregate.get("composite", 0.0)) < args.fail_under:
        message = (
            f"composite {aggregate.get('composite', 0.0):.4f} "
            f"below threshold {args.fail_under:.4f}"
        )
        print(message, file=sys.stderr)
        return 2
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    golden = filter_examples(
        load_examples(args.golden),
        language=args.language,
        severity=args.severity,
        tag=args.tag,
    )
    pred_a = load_predictions(args.predictions_a)
    pred_b = load_predictions(args.predictions_b)
    agg_a, _ = evaluate_predictions(golden, pred_a)
    agg_b, _ = evaluate_predictions(golden, pred_b)
    keys = [
        "composite",
        "token_f1",
        "bleu_lite",
        "rouge_l",
        "must_mention_recall",
        "severity_accuracy",
        "tag_f1",
        "forbidden_rate",
    ]
    delta = {
        key: round(float(agg_b.get(key, 0.0)) - float(agg_a.get(key, 0.0)), 6)  # type: ignore[arg-type]
        for key in keys
        if key in agg_a and key in agg_b
    }
    report = {"a": agg_a, "b": agg_b, "delta_b_minus_a": delta, "count": len(golden)}
    if args.out:
        _write_json(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    golden = load_examples(args.golden)
    predictions = [heuristic_prediction(example) for example in golden]
    write_jsonl(
        args.out,
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
    print(f"wrote {len(predictions)} baseline predictions to {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    try:
        examples = load_examples(args.path)
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1
    # load_examples already validates rows; report summary
    ids = [ex.id for ex in examples]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids detected")
    for ex in examples:
        if not ex.diff.strip():
            errors.append(f"{ex.id}: empty diff")
        if not ex.target_comment.strip():
            errors.append(f"{ex.id}: empty comment")
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "count": len(examples), "path": args.path}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-eval",
        description="Evaluate code-review model predictions against a golden set.",
    )
    parser.add_argument("--version", action="version", version=f"review-tuner {__version__}")
    sub = parser.add_subparsers(dest="command")

    # Default eval (also as subcommand for clarity)
    eval_p = sub.add_parser("eval", help="Score predictions against golden JSONL.")
    _add_eval_args(eval_p)
    eval_p.set_defaults(func=cmd_eval)

    cmp_p = sub.add_parser("compare", help="Compare two prediction files on the same golden set.")
    cmp_p.add_argument("--golden", required=True)
    cmp_p.add_argument("predictions_a")
    cmp_p.add_argument("predictions_b")
    cmp_p.add_argument("--out", default=None)
    cmp_p.add_argument("--language", default=None)
    cmp_p.add_argument("--severity", default=None)
    cmp_p.add_argument("--tag", default=None)
    cmp_p.set_defaults(func=cmd_compare)

    base_p = sub.add_parser("baseline", help="Write deterministic heuristic predictions.")
    base_p.add_argument("--golden", required=True)
    base_p.add_argument("--out", required=True)
    base_p.set_defaults(func=cmd_baseline)

    val_p = sub.add_parser("validate", help="Validate a JSONL dataset schema.")
    val_p.add_argument("path")
    val_p.set_defaults(func=cmd_validate)

    # Backward-compatible top-level flags (no subcommand) → eval
    _add_eval_args(parser)
    return parser


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--golden", help="Path to golden JSONL examples.")
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
        "--report-md",
        default=None,
        help="Optional Markdown report path.",
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
    parser.add_argument("--language", default=None, help="Filter golden examples by language.")
    parser.add_argument("--severity", default=None, help="Filter golden examples by severity.")
    parser.add_argument("--tag", default=None, help="Filter golden examples by tag.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # If first arg is not a known subcommand and looks like a flag, treat as eval.
    known = {"eval", "compare", "baseline", "validate", "-h", "--help", "--version"}
    if argv and argv[0] not in known and argv[0].startswith("-"):
        argv = ["eval", *argv]
    elif not argv:
        argv = ["eval", "--help"]
    elif argv[0] not in known and not argv[0].startswith("-"):
        # allow legacy: review-eval --golden ... without subcommand
        if any(a.startswith("--") for a in argv):
            argv = ["eval", *argv]

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        # top-level eval flags without subcommand name after parse quirks
        if getattr(args, "golden", None):
            return cmd_eval(args)
        parser.print_help()
        return 2
    try:
        # eval requires --golden
        if args.func is cmd_eval and not getattr(args, "golden", None):
            print("error: --golden is required", file=sys.stderr)
            return 2
        return int(args.func(args))
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
