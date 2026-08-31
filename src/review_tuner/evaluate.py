"""Command line evaluation harness for golden code review datasets."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from review_tuner import __version__
from review_tuner.data import index_by_id, load_examples, load_predictions, write_jsonl
from review_tuner.lint import lint_dataset
from review_tuner.metrics import (
    ExampleScore,
    aggregate_scores,
    render_html_report,
    render_markdown_report,
    score_example,
)
from review_tuner.schema import DatasetError, Prediction, ReviewExample
from review_tuner.view import build_eval_view, unmatched_prediction_ids

_DELETED_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DELETED_STOPWORDS = {
    "a",
    "an",
    "and",
    "const",
    "def",
    "else",
    "false",
    "fn",
    "for",
    "from",
    "func",
    "if",
    "import",
    "in",
    "let",
    "nil",
    "none",
    "not",
    "null",
    "or",
    "return",
    "self",
    "the",
    "this",
    "true",
    "var",
}


def deleted_line_tokens(diff: str) -> list[str]:
    """Identifiers from deleted diff lines, excluding unified-diff `---` headers."""

    tokens: list[str] = []
    seen: set[str] = set()
    for raw_line in diff.splitlines():
        if not raw_line.startswith("-") or raw_line.startswith("---"):
            continue
        for token in _DELETED_IDENT.findall(raw_line[1:]):
            key = token.lower()
            if len(token) < 2 or key in seen or key in _DELETED_STOPWORDS:
                continue
            seen.add(key)
            tokens.append(token)
    return tokens


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

    removed = deleted_line_tokens(example.diff)
    if removed:
        shown = ", ".join(removed[:16])
        comment = f"{comment} Removed lines mention: {shown}."
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


def score_predictions(
    golden: list[ReviewExample], predictions: list[Prediction]
) -> list[ExampleScore]:
    """Score each golden example; require a prediction for every id."""

    prediction_by_id = index_by_id(predictions)
    missing = [example.id for example in golden if example.id not in prediction_by_id]
    if missing:
        raise DatasetError(f"missing predictions for ids: {', '.join(missing)}")
    return [score_example(example, prediction_by_id[example.id]) for example in golden]


def evaluate_predictions(
    golden: list[ReviewExample], predictions: list[Prediction]
) -> tuple[dict, list[dict]]:
    """Evaluate predictions against golden examples."""

    scores = score_predictions(golden, predictions)
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

    extra = unmatched_prediction_ids(golden, predictions)
    view = build_eval_view(
        golden,
        predictions,
        extra_prediction_ids=extra,
        threshold=float(getattr(args, "fail_under", None) or 0.60),
    )
    aggregate = view["aggregate"]
    analysis = view["error_analysis"]
    include_text = bool(getattr(args, "include_text", False))
    per_example = view["examples"] if include_text else [
        {key: row[key] for key in row if key not in {
            "diff",
            "expected_comment",
            "prediction",
            "context",
            "file_path",
            "explanation",
            "must_mention",
            "avoid",
            "must_mention_hits",
            "tags",
            "predicted_tags",
        }}
        for row in view["examples"]
    ]
    payload = dict(aggregate)
    payload["error_analysis"] = analysis
    payload["letter_grade"] = view["letter_grade"]
    payload["story"] = view["story"]
    if extra:
        payload["extra_prediction_ids"] = extra

    out_path = Path(args.out)
    _write_json(out_path, payload)
    write_jsonl(args.per_example_out, per_example)
    report_json = getattr(args, "report_json", None)
    if report_json:
        _write_json(Path(report_json), payload)
    if args.report_md:
        md_path = Path(args.report_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            render_markdown_report(aggregate, analysis=analysis, per_example=per_example),
            encoding="utf-8",
        )
    report_html = getattr(args, "report_html", None)
    if report_html:
        html_path = Path(report_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            render_html_report(aggregate, view["examples"], analysis),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))

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


def cmd_lint(args: argparse.Namespace) -> int:
    issues = lint_dataset(args.path)
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]
    for issue in issues:
        stream = sys.stderr if issue.level == "error" else sys.stdout
        print(issue.format(), file=stream)
    summary = {
        "ok": not errors,
        "errors": len(errors),
        "warnings": len(warnings),
        "path": args.path,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


def _load_eval_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and "composite" in payload:
        return payload
    return None


def render_dashboard(reports_dir: str | Path) -> str:
    """Build a static index summarizing the newest eval JSON/Markdown reports."""

    root = Path(reports_dir)
    json_reports: list[tuple[float, Path, dict]] = []
    for path in root.glob("*.json"):
        payload = _load_eval_json(path)
        if payload is None:
            continue
        json_reports.append((path.stat().st_mtime, path, payload))
    json_reports.sort(key=lambda item: item[0], reverse=True)

    md_reports = sorted(
        (path for path in root.glob("*.md")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    html_reports = sorted(
        (path for path in root.glob("*.html") if path.name != "index.html"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    latest_block = "<p class='empty'>No eval JSON reports found. Re-run with --report-json.</p>"
    if json_reports:
        _, latest_path, latest = json_reports[0]
        count = int(latest.get("count", 0))
        composite = float(latest.get("composite", 0.0) or 0.0)
        ci_lo = latest.get("composite_ci_lo")
        ci_hi = latest.get("composite_ci_hi")
        ci_line = ""
        if ci_lo is not None and ci_hi is not None:
            ci_line = f"<p>Composite 95% CI {float(ci_lo):.4f} – {float(ci_hi):.4f}</p>"
        generated = datetime.fromtimestamp(latest_path.stat().st_mtime, tz=timezone.utc)
        latest_block = (
            f"<p class='score'>composite {composite:.4f}</p>"
            f"<p><strong>{html.escape(latest_path.name)}</strong> · {count} examples · "
            f"{generated.strftime('%Y-%m-%d %H:%M UTC')}</p>"
            f"{ci_line}"
        )

    def _file_list(paths: list[Path], empty: str) -> str:
        if not paths:
            return f"<p class='empty'>{html.escape(empty)}</p>"
        items = "".join(
            f"<li><a href='{html.escape(path.name)}'>{html.escape(path.name)}</a></li>"
            for path in paths
        )
        return f"<ul>{items}</ul>"

    json_items = _file_list([path for _, path, _ in json_reports], "No JSON eval reports.")
    md_items = _file_list(md_reports, "No Markdown reports.")
    html_items = _file_list(html_reports, "No HTML reports.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Review eval dashboard</title>
<style>
body {{
  margin: 0;
  color: #12202b;
  background: #f6f8f7;
  font: 15px/1.5 "Segoe UI", system-ui, sans-serif;
}}
header {{ background: #12202b; color: #f4f7f6; padding: 1.5rem 1.3rem; }}
h1 {{ margin: 0 0 0.4rem; font-size: 1.5rem; }}
main {{ max-width: 880px; margin: 0 auto; padding: 1.3rem; }}
section {{
  background: #fff;
  border: 1px solid #d7e0e6;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
}}
.score {{ font-size: 2rem; font-weight: 700; color: #0f6f62; margin: 0.2rem 0; }}
.empty {{ color: #5b6b75; }}
a {{ color: #0f6f62; }}
</style>
</head>
<body>
<header>
  <h1>Review eval dashboard</h1>
  <p>Latest reports in {html.escape(str(root))}</p>
</header>
<main>
<section>
  <h2>Latest eval</h2>
  {latest_block}
</section>
<section>
  <h2>JSON reports</h2>
  {json_items}
</section>
<section>
  <h2>Markdown reports</h2>
  {md_items}
</section>
<section>
  <h2>HTML reports</h2>
  {html_items}
</section>
</main>
</body>
</html>
"""


def cmd_studio(args: argparse.Namespace) -> int:
    from review_tuner.studio import main as studio_main

    argv = ["--host", str(args.host), "--port", str(args.port)]
    if getattr(args, "open_browser", False):
        argv.append("--open")
    return studio_main(argv)


def cmd_dashboard(args: argparse.Namespace) -> int:
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_dir():
        print(f"error: not a directory: {reports_dir}", file=sys.stderr)
        return 1
    html_path = reports_dir / "index.html"
    html_path.write_text(render_dashboard(reports_dir), encoding="utf-8")
    print(f"wrote {html_path}")
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

    lint_p = sub.add_parser("lint", help="Lint a golden or training JSONL dataset.")
    lint_p.add_argument("path")
    lint_p.set_defaults(func=cmd_lint)

    dash_p = sub.add_parser("dashboard", help="Write a static HTML index of eval reports.")
    dash_p.add_argument("reports_dir")
    dash_p.set_defaults(func=cmd_dashboard)

    studio_p = sub.add_parser("studio", help="Open the local Review Tuner Studio in a browser.")
    studio_p.add_argument("--host", default="127.0.0.1")
    studio_p.add_argument("--port", type=int, default=8765)
    studio_p.add_argument("--open", action="store_true", dest="open_browser")
    studio_p.set_defaults(func=cmd_studio)

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
        "--report-html",
        default=None,
        help="Optional self-contained HTML report path.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON report path (aggregate metrics plus error analysis).",
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
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include diffs and comments in the per-example JSONL (always on for HTML).",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # If first arg is not a known subcommand and looks like a flag, treat as eval.
    known = {
        "eval",
        "compare",
        "baseline",
        "validate",
        "lint",
        "dashboard",
        "studio",
        "-h",
        "--help",
        "--version",
    }
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
