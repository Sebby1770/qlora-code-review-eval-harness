"""Markdown/HTML reports, slices, letter grades, and bootstrap intervals."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from review_tuner.data import (
    index_by_id,
    iter_examples,
    iter_jsonl,
    iter_predictions,
)
from review_tuner.metrics import (
    ExampleScore,
    ExtendedExampleScore,
    aggregate_scores,
    extend_score,
)
from review_tuner.schema import DatasetError, Prediction, ReviewExample

DEFAULT_THRESHOLD = 0.60
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 1337
WEAK_MARGIN = 0.10

_CORE_SCORE_KEYS = (
    "exact_match",
    "token_f1",
    "must_mention_recall",
    "forbidden_rate",
    "severity_accuracy",
    "tag_f1",
    "composite",
)


def letter_grade(composite: float) -> str:
    """Map a unit-interval composite onto an A–F scale."""

    if composite >= 0.90:
        return "A"
    if composite >= 0.80:
        return "B"
    if composite >= 0.70:
        return "C"
    if composite >= 0.60:
        return "D"
    return "F"


def gate_status(composite: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Classify a run as pass, weak, or fail against a composite threshold."""

    if composite >= threshold:
        return "pass"
    if composite >= threshold - WEAK_MARGIN:
        return "weak"
    return "fail"


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Return a deterministic 95% bootstrap CI on the mean, or empty if n < 3."""

    if len(values) < 3:
        return {}
    rng = random.Random(seed)
    count = len(values)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return {
        "low": means[low_index],
        "high": means[high_index],
        "samples": samples,
        "seed": seed,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def slice_composites(
    examples: Sequence[ReviewExample],
    scores: Sequence[ExampleScore],
) -> dict[str, dict[str, float]]:
    """Mean composite grouped by language, severity, and tag."""

    if len(examples) != len(scores):
        raise ValueError("examples and scores must be aligned")

    by_language: dict[str, list[float]] = defaultdict(list)
    by_severity: dict[str, list[float]] = defaultdict(list)
    by_tag: dict[str, list[float]] = defaultdict(list)
    for example, score in zip(examples, scores, strict=True):
        by_language[example.language].append(score.composite)
        by_severity[example.severity].append(score.composite)
        for tag in example.tags:
            by_tag[tag].append(score.composite)

    def _sorted_means(groups: Mapping[str, list[float]]) -> dict[str, float]:
        return {name: _mean(values) for name, values in sorted(groups.items())}

    return {
        "by_language": _sorted_means(by_language),
        "by_severity": _sorted_means(by_severity),
        "by_tag": _sorted_means(by_tag),
    }


def example_detail(
    example: ReviewExample,
    prediction: Prediction,
    score: ExampleScore | None = None,
) -> dict[str, Any]:
    """Build a report row with additive metrics and inspector fields."""

    extended = extend_score(example, prediction, score)
    return detail_from_extended(example, prediction, extended)


def detail_from_extended(
    example: ReviewExample,
    prediction: Prediction,
    extended: ExtendedExampleScore,
) -> dict[str, Any]:
    """Serialize an extended score plus the original example/prediction text."""

    row: dict[str, Any] = dict(extended.as_dict())
    row.update(
        {
            "language": example.language,
            "file_path": example.file_path,
            "severity_expected": example.severity,
            "tags": list(example.tags),
            "diff": example.diff,
            "expected_comment": example.target_comment,
            "prediction": prediction.prediction,
            "predicted_severity": prediction.severity,
            "predicted_tags": list(prediction.tags),
            "bleu_lite": extended.bleu_lite,
            "rouge_l_lite": extended.rouge_l_lite,
            "length_ratio": extended.length_ratio,
            "security_fail": extended.security_fail,
            "missed_must_mention": list(extended.missed_must_mention),
            "hit_avoid": list(extended.hit_avoid),
        }
    )
    return row


def english_summary(
    aggregate: Mapping[str, Any],
    examples: Sequence[ReviewExample],
    scores: Sequence[ExampleScore],
) -> str:
    """One paragraph a reviewer can paste into a PR."""

    composite = float(aggregate.get("composite", 0.0))
    grade = str(aggregate.get("letter_grade") or letter_grade(composite))
    threshold = float(aggregate.get("threshold", DEFAULT_THRESHOLD))
    gate = str(aggregate.get("gate") or gate_status(composite, threshold))
    count = int(aggregate.get("count", len(scores)))
    security_rate = float(aggregate.get("security_fail_rate", 0.0))
    parts = [
        (
            f"Grade {grade} with mean composite {composite:.3f} on {count} examples "
            f"({gate} vs threshold {threshold:.2f})."
        )
    ]
    if security_rate > 0.0:
        parts.append(
            f"Security-fail rate is {security_rate:.0%}: at least one blocker/high "
            "case missed a required phrase."
        )
    else:
        parts.append("All blocker/high cases mentioned every required phrase.")
    if examples and scores:
        ranked = sorted(
            zip(examples, scores, strict=True),
            key=lambda item: item[1].composite,
        )
        weakest = ", ".join(
            f"{example.id} ({example.language}/{example.severity}, "
            f"{score.composite:.3f})"
            for example, score in ranked[:3]
        )
        parts.append(f"Weakest examples: {weakest}.")
    slices = aggregate.get("slices")
    if isinstance(slices, dict):
        by_language = slices.get("by_language")
        if isinstance(by_language, dict) and by_language:
            worst_language, worst_value = min(
                by_language.items(), key=lambda item: float(item[1])
            )
            parts.append(
                f"Lowest language slice is {worst_language} at {float(worst_value):.3f}."
            )
    return " ".join(parts)


def build_eval_report(
    examples: Sequence[ReviewExample],
    predictions: Sequence[Prediction],
    scores: Sequence[ExampleScore],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Extend the public aggregate dict with slices, CI, grade, and extras."""

    if len(examples) != len(scores):
        raise ValueError("examples and scores must be aligned")
    aggregate: dict[str, Any] = dict(aggregate_scores(scores))
    if not scores:
        return aggregate

    prediction_by_id = index_by_id(predictions)
    extras = {"bleu_lite": 0.0, "rouge_l_lite": 0.0, "length_ratio": 0.0, "security_fail": 0.0}
    for example, score in zip(examples, scores, strict=True):
        prediction = prediction_by_id[example.id]
        extended = extend_score(example, prediction, score)
        extras["bleu_lite"] += extended.bleu_lite
        extras["rouge_l_lite"] += extended.rouge_l_lite
        extras["length_ratio"] += extended.length_ratio
        extras["security_fail"] += extended.security_fail
    count = len(scores)
    aggregate["bleu_lite"] = extras["bleu_lite"] / count
    aggregate["rouge_l_lite"] = extras["rouge_l_lite"] / count
    aggregate["length_ratio"] = extras["length_ratio"] / count
    aggregate["security_fail_rate"] = extras["security_fail"] / count
    aggregate["slices"] = slice_composites(examples, scores)
    interval = bootstrap_ci([score.composite for score in scores])
    if interval:
        aggregate["bootstrap_ci"] = interval
    composite = float(aggregate["composite"])
    aggregate["letter_grade"] = letter_grade(composite)
    aggregate["gate"] = gate_status(composite, threshold)
    aggregate["threshold"] = threshold
    aggregate["summary"] = english_summary(aggregate, examples, scores)
    return aggregate


def build_details(
    examples: Sequence[ReviewExample],
    predictions: Sequence[Prediction],
    scores: Sequence[ExampleScore],
) -> list[dict[str, Any]]:
    """Per-example inspector rows in golden-set order."""

    prediction_by_id = index_by_id(predictions)
    return [
        example_detail(example, prediction_by_id[example.id], score)
        for example, score in zip(examples, scores, strict=True)
    ]


def write_text_report(path: str | Path, body: str) -> None:
    """Write a UTF-8 text report, creating parent directories as needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else f"{body}\n"
    destination.write_text(text, encoding="utf-8")


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def render_markdown(
    aggregate: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
) -> str:
    """Render a GitHub-flavored Markdown evaluation report."""

    composite = float(aggregate.get("composite", 0.0))
    grade = str(aggregate.get("letter_grade") or letter_grade(composite))
    gate = str(aggregate.get("gate") or gate_status(composite))
    threshold = float(aggregate.get("threshold", DEFAULT_THRESHOLD))
    lines = [
        "# Code review evaluation",
        "",
        f"**Grade {grade}** · composite `{composite:.4f}` · **{gate}** "
        f"(threshold `{threshold:.2f}`)",
        "",
    ]
    summary = aggregate.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.extend([summary.strip(), ""])

    lines.extend(["## Aggregate", "", "| Metric | Mean | Std |", "| --- | ---: | ---: |"])
    for key in _CORE_SCORE_KEYS:
        mean = aggregate.get(key)
        std = aggregate.get(f"{key}_std", "")
        if mean is None:
            continue
        lines.append(f"| `{key}` | {_fmt(mean)} | {_fmt(std) if std != '' else ''} |")
    for key in ("bleu_lite", "rouge_l_lite", "length_ratio", "security_fail_rate"):
        if key in aggregate:
            lines.append(f"| `{key}` | {_fmt(aggregate[key])} | |")
    lines.append(f"| `count` | {aggregate.get('count', len(details))} | |")
    lines.append("")

    interval = aggregate.get("bootstrap_ci")
    if isinstance(interval, dict) and interval:
        lines.extend(
            [
                "## Bootstrap 95% CI (composite)",
                "",
                f"- low: `{_fmt(interval.get('low'))}`",
                f"- high: `{_fmt(interval.get('high'))}`",
                f"- samples: `{interval.get('samples', BOOTSTRAP_SAMPLES)}`",
                f"- seed: `{interval.get('seed', BOOTSTRAP_SEED)}`",
                "",
            ]
        )

    slices = aggregate.get("slices")
    if isinstance(slices, dict) and slices:
        lines.extend(["## Slices", ""])
        for slice_name in ("by_language", "by_severity", "by_tag"):
            group = slices.get(slice_name)
            if not isinstance(group, dict) or not group:
                continue
            heading = slice_name.replace("by_", "").replace("_", " ")
            lines.extend(
                [
                    f"### {heading}",
                    "",
                    "| Key | Mean composite |",
                    "| --- | ---: |",
                ]
            )
            for name, value in group.items():
                lines.append(f"| `{_md_escape(str(name))}` | {_fmt(value)} |")
            lines.append("")

    if details:
        ranked = sorted(details, key=lambda row: float(row.get("composite", 0.0)))
        lines.extend(["## Examples (worst first)", ""])
        for row in ranked:
            example_id = str(row.get("id", ""))
            lines.append(
                f"### `{_md_escape(example_id)}` — composite {_fmt(row.get('composite', 0.0))}"
            )
            lines.append("")
            meta = []
            if row.get("language"):
                meta.append(f"language `{row['language']}`")
            if row.get("severity_expected"):
                meta.append(f"severity `{row['severity_expected']}`")
            if row.get("file_path"):
                meta.append(f"`{row['file_path']}`")
            if row.get("security_fail"):
                meta.append("security-fail")
            if meta:
                lines.append(" · ".join(meta))
                lines.append("")
            missed = row.get("missed_must_mention") or []
            if missed:
                joined = ", ".join(f"`{_md_escape(str(item))}`" for item in missed)
                lines.append(f"Missed must-mention: {joined}")
                lines.append("")
            if row.get("diff"):
                lines.extend(["Expected diff:", "", "```diff", str(row["diff"]), "```", ""])
            if row.get("expected_comment"):
                lines.extend(["Expected comment:", "", str(row["expected_comment"]), ""])
            if row.get("prediction"):
                lines.extend(["Prediction:", "", str(row["prediction"]), ""])
    return "\n".join(lines).rstrip() + "\n"


def _diff_html(diff: str) -> str:
    rendered: list[str] = []
    for line in diff.splitlines():
        css_class = "ctx"
        if line.startswith(("+++", "---")):
            css_class = "file"
        elif line.startswith("+"):
            css_class = "add"
        elif line.startswith("-"):
            css_class = "del"
        elif line.startswith("@@"):
            css_class = "hunk"
        rendered.append(f'<span class="d {css_class}">{html.escape(line)}</span>')
    return '<pre class="diff">' + "\n".join(rendered) + "</pre>"


def _prose_html(title: str, body: str) -> str:
    return (
        f'<div class="pane"><h4>{html.escape(title)}</h4>'
        f"<pre>{html.escape(body)}</pre></div>"
    )


def render_html(
    aggregate: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
) -> str:
    """Render a self-contained dark HTML report. No JavaScript required."""

    composite = float(aggregate.get("composite", 0.0))
    grade = str(aggregate.get("letter_grade") or letter_grade(composite))
    gate = str(aggregate.get("gate") or gate_status(composite))
    threshold = float(aggregate.get("threshold", DEFAULT_THRESHOLD))
    summary = str(aggregate.get("summary") or "")
    count = aggregate.get("count", len(details))

    metric_rows: list[str] = []
    for key in _CORE_SCORE_KEYS:
        if key not in aggregate:
            continue
        std = aggregate.get(f"{key}_std", "")
        metric_rows.append(
            "<tr>"
            f"<th>{html.escape(key)}</th>"
            f"<td>{html.escape(_fmt(aggregate[key]))}</td>"
            f"<td>{html.escape(_fmt(std) if std != '' else '—')}</td>"
            "</tr>"
        )
    for key in ("bleu_lite", "rouge_l_lite", "length_ratio", "security_fail_rate"):
        if key in aggregate:
            metric_rows.append(
                "<tr>"
                f"<th>{html.escape(key)}</th>"
                f"<td>{html.escape(_fmt(aggregate[key]))}</td>"
                "<td>—</td>"
                "</tr>"
            )

    slice_html: list[str] = []
    slices = aggregate.get("slices")
    if isinstance(slices, dict):
        for slice_name in ("by_language", "by_severity", "by_tag"):
            group = slices.get(slice_name)
            if not isinstance(group, dict) or not group:
                continue
            items = "".join(
                f"<li><code>{html.escape(str(name))}</code> "
                f"<span>{html.escape(_fmt(value))}</span></li>"
                for name, value in group.items()
            )
            heading = slice_name.replace("by_", "").replace("_", " ")
            slice_html.append(f"<section><h3>{html.escape(heading)}</h3><ul>{items}</ul></section>")

    interval = aggregate.get("bootstrap_ci")
    ci_html = ""
    if isinstance(interval, dict) and interval:
        ci_html = (
            "<p class='ci'>Bootstrap 95% CI on composite: "
            f"<code>{html.escape(_fmt(interval.get('low')))}</code> – "
            f"<code>{html.escape(_fmt(interval.get('high')))}</code> "
            f"(n={html.escape(str(interval.get('samples', BOOTSTRAP_SAMPLES)))}, "
            f"seed={html.escape(str(interval.get('seed', BOOTSTRAP_SEED)))})</p>"
        )

    ranked = sorted(details, key=lambda row: float(row.get("composite", 0.0)))
    example_rows: list[str] = []
    for row in ranked:
        missed = row.get("missed_must_mention") or []
        missed_html = (
            "<p class='missed'><strong>Missed must-mention:</strong> "
            + ", ".join(f"<code>{html.escape(str(item))}</code>" for item in missed)
            + "</p>"
            if missed
            else "<p class='ok'>All required phrases mentioned.</p>"
        )
        diff = str(row["diff"]) if row.get("diff") else ""
        expected = str(row["expected_comment"]) if row.get("expected_comment") else ""
        prediction = str(row["prediction"]) if row.get("prediction") else ""
        inspector = []
        if diff:
            inspector.append(_diff_html(diff))
        inspector.append('<div class="split">')
        inspector.append(_prose_html("Expected", expected or "(not provided)"))
        inspector.append(_prose_html("Prediction", prediction or "(not provided)"))
        inspector.append("</div>")
        inspector.append(missed_html)
        extra_bits = []
        for key in ("bleu_lite", "rouge_l_lite", "length_ratio", "security_fail"):
            if key in row:
                extra_bits.append(f"{key}={_fmt(row[key])}")
        if extra_bits:
            inspector.append(f"<p class='extra'>{html.escape(' · '.join(extra_bits))}</p>")
        meta = " ".join(
            part
            for part in (
                str(row.get("language") or ""),
                str(row.get("severity_expected") or ""),
                str(row.get("file_path") or ""),
            )
            if part
        )
        fail_class = " security-fail" if row.get("security_fail") else ""
        example_rows.append(
            "<tr class='example{fail}'>"
            "<td><code>{id}</code><div class='meta'>{meta}</div></td>"
            "<td>{composite}</td>"
            "<td>{token}</td>"
            "<td>{mention}</td>"
            "<td>{severity}</td>"
            "<td>{tags}</td>"
            "<td><details><summary>Inspect</summary>{body}</details></td>"
            "</tr>".format(
                fail=fail_class,
                id=html.escape(str(row.get("id", ""))),
                meta=html.escape(meta),
                composite=html.escape(_fmt(row.get("composite", 0.0))),
                token=html.escape(_fmt(row.get("token_f1", 0.0))),
                mention=html.escape(_fmt(row.get("must_mention_recall", 0.0))),
                severity=html.escape(_fmt(row.get("severity_accuracy", 0.0))),
                tags=html.escape(_fmt(row.get("tag_f1", 0.0))),
                body="".join(inspector),
            )
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Code review eval — grade {html.escape(grade)}</title>
  <style>
    :root {{
      --bg: #0e1117;
      --panel: #161b22;
      --ink: #e6edf3;
      --muted: #8b949e;
      --line: #30363d;
      --add: #3fb950;
      --del: #f85149;
      --hunk: #79c0ff;
      --fail: #f85149;
      --pass: #3fb950;
      --weak: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 ui-sans-serif, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1, h2, h3, h4 {{ font-weight: 650; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    p {{ color: var(--muted); }}
    .hero {{
      display: flex; gap: 24px; align-items: center;
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 20px 24px;
    }}
    .grade {{
      font-size: 64px; font-weight: 750; line-height: 1;
      min-width: 72px; text-align: center;
    }}
    .grade.A {{ color: var(--pass); }}
    .grade.B {{ color: #56d364; }}
    .grade.C {{ color: var(--weak); }}
    .grade.D {{ color: #e3b341; }}
    .grade.F {{ color: var(--fail); }}
    .chip {{
      display: inline-block; border-radius: 999px; padding: 2px 10px;
      font-size: 12px; font-weight: 650; letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .chip.pass {{ background: #0f3d24; color: var(--pass); }}
    .chip.weak {{ background: #3d2f0f; color: var(--weak); }}
    .chip.fail {{ background: #3d1010; color: var(--fail); }}
    table {{
      width: 100%; border-collapse: collapse; background: var(--panel);
      border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    }}
    th, td {{ padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
    td {{ border-top: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .slices {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
    }}
    .slices section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 16px;
    }}
    .slices ul {{ list-style: none; margin: 0; padding: 0; }}
    .slices li {{ display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    details {{
      background: #0e1117; border: 1px solid var(--line); border-radius: 8px; padding: 8px;
    }}
    summary {{ cursor: pointer; color: var(--hunk); }}
    .diff {{
      margin: 8px 0; padding: 10px; overflow: auto; background: #010409;
      border-radius: 8px; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .d {{ display: block; white-space: pre-wrap; }}
    .d.add {{ color: var(--add); }}
    .d.del {{ color: var(--del); }}
    .d.hunk {{ color: var(--hunk); }}
    .d.file {{ color: var(--muted); }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .pane {{ background: #010409; border-radius: 8px; padding: 8px 10px; }}
    .pane h4 {{ margin: 0 0 6px; font-size: 12px; color: var(--muted); }}
    .pane pre {{
      white-space: pre-wrap; margin: 0;
      font: 12px/1.45 ui-monospace, Menlo, Consolas, monospace;
    }}
    .missed {{ color: var(--fail); }}
    .ok {{ color: var(--pass); }}
    .security-fail {{ background: #2a1215; }}
    @media (max-width: 800px) {{ .split {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <div class="grade {html.escape(grade)}">{html.escape(grade)}</div>
    <div>
      <h1>Code review evaluation</h1>
      <p>composite <code>{html.escape(_fmt(composite))}</code>
      · {html.escape(str(count))} examples ·
      threshold <code>{html.escape(_fmt(threshold, 2))}</code>
      <span class="chip {html.escape(gate)}">{html.escape(gate)}</span></p>
      <p>{html.escape(summary)}</p>
      {ci_html}
    </div>
  </div>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Mean</th><th>Std</th></tr></thead>
    <tbody>
      {''.join(metric_rows)}
    </tbody>
  </table>
  <h2>Slices</h2>
  <div class="slices">{''.join(slice_html) or '<p>No slice metadata.</p>'}</div>
  <h2>Examples</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Composite</th><th>Token F1</th><th>Must-mention</th>
        <th>Severity</th><th>Tag F1</th><th>Details</th>
      </tr>
    </thead>
    <tbody>
      {''.join(example_rows) or '<tr><td colspan="7">No examples.</td></tr>'}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def _score_from_record(record: Mapping[str, Any]) -> ExampleScore:
    def _field(name: str) -> float:
        value = record.get(name, 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        raise DatasetError(f"per-example record missing numeric field {name}")

    example_id = record.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise DatasetError("per-example record is missing id")
    return ExampleScore(
        id=example_id,
        exact_match=_field("exact_match"),
        token_f1=_field("token_f1"),
        must_mention_recall=_field("must_mention_recall"),
        forbidden_rate=_field("forbidden_rate"),
        severity_accuracy=_field("severity_accuracy"),
        tag_f1=_field("tag_f1"),
        composite=_field("composite"),
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"could not read eval json {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DatasetError(f"{source}: expected a JSON object")
    return loaded


def format_slices(slices: Mapping[str, Mapping[str, float]]) -> str:
    """Pretty-print slice tables for the CLI."""

    blocks: list[str] = []
    for slice_name in ("by_language", "by_severity", "by_tag"):
        group = slices.get(slice_name) or {}
        heading = slice_name.replace("_", " ")
        blocks.append(heading)
        if not group:
            blocks.append("  (none)")
            continue
        width = max(len(str(name)) for name in group)
        for name, value in group.items():
            blocks.append(f"  {str(name):<{width}}  {value:.4f}")
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def slices_main(argv: list[str] | None = None) -> int:
    """Print composite means by language, severity, and tag."""

    parser = argparse.ArgumentParser(
        prog="review-eval slices",
        description="Print mean composite by language, severity, and tag.",
    )
    parser.add_argument("--golden", required=True, help="Path to golden JSONL examples.")
    parser.add_argument(
        "--predictions",
        help="Path to prediction JSONL. If omitted, the heuristic baseline is used.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable slices.")
    args = parser.parse_args(argv)
    try:
        from review_tuner.evaluate import heuristic_prediction, iter_scores

        examples = list(iter_examples(args.golden))
        if args.predictions:
            predictions = list(iter_predictions(args.predictions))
        else:
            predictions = [heuristic_prediction(example) for example in examples]
        scores = list(iter_scores(examples, predictions))
        slices = slice_composites(examples, scores)
        if args.json:
            print(json.dumps(slices, indent=2, sort_keys=True))
        else:
            print(format_slices(slices), end="")
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


def report_main(argv: list[str] | None = None) -> int:
    """Regenerate Markdown/HTML from an eval JSON and optional source files."""

    parser = argparse.ArgumentParser(
        prog="review-eval report",
        description="Render Markdown/HTML from an eval JSON plus per-example scores.",
    )
    parser.add_argument("--eval", required=True, help="Aggregate eval JSON from review-eval.")
    parser.add_argument(
        "--examples",
        required=True,
        help="Per-example score JSONL from review-eval.",
    )
    parser.add_argument("--golden", help="Golden JSONL used to attach diffs and comments.")
    parser.add_argument("--predictions", help="Prediction JSONL used to attach model text.")
    parser.add_argument("--report-md", help="Markdown output path.")
    parser.add_argument("--report-html", help="HTML output path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Pass/weak/fail threshold if the JSON does not already include a gate.",
    )
    args = parser.parse_args(argv)
    try:
        if not args.report_md and not args.report_html:
            raise DatasetError("provide --report-md and/or --report-html")
        if not 0.0 <= args.threshold <= 1.0:
            raise DatasetError("--threshold must be between 0 and 1")
        aggregate = _load_json(args.eval)
        score_records = list(iter_jsonl(args.examples))
        scores = [_score_from_record(record) for record in score_records]
        details: list[dict[str, Any]] = [dict(record) for record in score_records]
        if args.golden:
            golden = list(iter_examples(args.golden))
            if len(golden) != len(scores):
                raise DatasetError("golden examples and per-example scores are different lengths")
            if args.predictions:
                predictions = list(iter_predictions(args.predictions))
            else:
                predictions = []
            if predictions:
                details = build_details(golden, predictions, scores)
            composite = float(aggregate.get("composite", 0.0))
            aggregate.setdefault("letter_grade", letter_grade(composite))
            aggregate.setdefault("threshold", args.threshold)
            aggregate.setdefault("gate", gate_status(composite, float(aggregate["threshold"])))
            if "slices" not in aggregate:
                aggregate["slices"] = slice_composites(golden, scores)
            if "summary" not in aggregate:
                aggregate["summary"] = english_summary(aggregate, golden, scores)
        else:
            composite = float(aggregate.get("composite", 0.0))
            aggregate.setdefault("letter_grade", letter_grade(composite))
            aggregate.setdefault("threshold", args.threshold)
            aggregate.setdefault("gate", gate_status(composite, float(aggregate["threshold"])))
        if args.report_md:
            write_text_report(args.report_md, render_markdown(aggregate, details))
        if args.report_html:
            write_text_report(args.report_html, render_html(aggregate, details))
        print(
            json.dumps(
                {
                    "letter_grade": aggregate.get("letter_grade"),
                    "gate": aggregate.get("gate"),
                    "composite": aggregate.get("composite"),
                    "markdown": args.report_md,
                    "html": args.report_html,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(report_main())
