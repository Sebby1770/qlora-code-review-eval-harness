"""Golden-set metrics for review comment generation."""

from __future__ import annotations

import html
import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from review_tuner.schema import VALID_SEVERITIES, Prediction, ReviewExample

_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_text(value: str) -> str:
    """Lowercase and collapse punctuation and whitespace."""

    return " ".join(value.lower().translate(_PUNCT_TRANSLATION).split())


def tokenize(value: str) -> list[str]:
    """Tokenize text for lightweight lexical metrics."""

    return normalize_text(value).split()


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def token_f1(prediction: str, target: str) -> float:
    """Compute bag-of-words token F1."""

    pred_tokens = tokenize(prediction)
    target_tokens = tokenize(target)
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum((pred_counts & target_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_lite(prediction: str, target: str, max_n: int = 2) -> float:
    """Tiny modified unigram+bigram precision BLEU without external deps."""

    pred = tokenize(prediction)
    ref = tokenize(target)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ng = Counter(ngrams(pred, n))
        ref_ng = Counter(ngrams(ref, n))
        if not pred_ng:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ng & ref_ng).values())
        precisions.append(overlap / sum(pred_ng.values()))
    # brevity penalty
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1.0 - len(ref) / max(len(pred), 1))
    if any(p <= 0 for p in precisions):
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bp * geo


def rouge_l_lite(prediction: str, target: str) -> float:
    """Character-token LCS F-measure (ROUGE-L style)."""

    pred = tokenize(prediction)
    ref = tokenize(target)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    # LCS length via DP
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, a in enumerate(pred, 1):
        for j, b in enumerate(ref, 1):
            if a == b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def length_ratio(prediction: str, target: str) -> float:
    """min(len)/max(len) on token counts; 1.0 if both empty."""

    p = len(tokenize(prediction))
    t = len(tokenize(target))
    if p == 0 and t == 0:
        return 1.0
    if p == 0 or t == 0:
        return 0.0
    return min(p, t) / max(p, t)


def phrase_recall(prediction: str, required_phrases: tuple[str, ...]) -> float:
    """Score the share of rubric phrases mentioned by the prediction."""

    if not required_phrases:
        return 1.0
    normalized = normalize_text(prediction)
    hits = sum(1 for phrase in required_phrases if normalize_text(phrase) in normalized)
    return hits / len(required_phrases)


def forbidden_rate(prediction: str, forbidden_phrases: tuple[str, ...]) -> float:
    """Score the share of forbidden phrases that appear in the prediction."""

    if not forbidden_phrases:
        return 0.0
    normalized = normalize_text(prediction)
    hits = sum(1 for phrase in forbidden_phrases if normalize_text(phrase) in normalized)
    return hits / len(forbidden_phrases)


def f1_for_sets(predicted: tuple[str, ...], expected: tuple[str, ...]) -> float:
    """Compute F1 for tag sets."""

    predicted_set = set(predicted)
    expected_set = set(expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    overlap = len(predicted_set & expected_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(expected_set)
    return 2 * precision * recall / (precision + recall)


def infer_severity(text: str) -> str | None:
    """Infer severity from structured or prose predictions."""

    normalized = normalize_text(text)
    match = re.search(r"\bseverity\s+(blocker|high|medium|low|nit)\b", normalized)
    if match:
        return match.group(1)
    for severity in VALID_SEVERITIES:
        if f"[{severity}]" in text.lower() or f"{severity} severity" in normalized:
            return severity
    return None


def missed_must_mention(prediction: str, required_phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Return rubric phrases that do not appear in the prediction."""

    normalized = normalize_text(prediction)
    return tuple(
        phrase for phrase in required_phrases if normalize_text(phrase) not in normalized
    )


def forbidden_hits(prediction: str, forbidden_phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Return forbidden phrases that appear in the prediction."""

    normalized = normalize_text(prediction)
    return tuple(phrase for phrase in forbidden_phrases if normalize_text(phrase) in normalized)


def _percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Linear-interpolation percentile for a sorted non-empty sequence."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percent / 100.0) * (len(sorted_values) - 1)
    low_index = int(math.floor(rank))
    high_index = int(math.ceil(rank))
    low = float(sorted_values[low_index])
    if low_index == high_index:
        return low
    high = float(sorted_values[high_index])
    frac = rank - low_index
    return low * (1.0 - frac) + high * frac


def bootstrap_ci(
    values: Sequence[float],
    n: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic 95% percentile bootstrap interval on the mean."""

    if n < 1:
        raise ValueError("n must be >= 1")
    if not values:
        return (0.0, 0.0)
    data = [float(value) for value in values]
    rng = random.Random(seed)
    size = len(data)
    means = []
    for _ in range(n):
        sample = [data[rng.randrange(size)] for _ in range(size)]
        means.append(statistics.fmean(sample))
    means.sort()
    return (_percentile(means, 2.5), _percentile(means, 97.5))


@dataclass(frozen=True)
class ExampleScore:
    """Metric bundle for one golden example."""

    id: str
    exact_match: float
    token_f1: float
    bleu_lite: float
    rouge_l: float
    length_ratio: float
    must_mention_recall: float
    forbidden_rate: float
    severity_accuracy: float
    tag_f1: float
    composite: float
    language: str = ""
    severity: str = ""
    predicted_severity: str = ""
    missed_must_mention: tuple[str, ...] = ()
    forbidden_hits: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, float | str | list[str]]:
        return {
            "id": self.id,
            "language": self.language,
            "severity": self.severity,
            "predicted_severity": self.predicted_severity,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "bleu_lite": self.bleu_lite,
            "rouge_l": self.rouge_l,
            "length_ratio": self.length_ratio,
            "must_mention_recall": self.must_mention_recall,
            "forbidden_rate": self.forbidden_rate,
            "severity_accuracy": self.severity_accuracy,
            "tag_f1": self.tag_f1,
            "composite": self.composite,
            "missed_must_mention": list(self.missed_must_mention),
            "forbidden_hits": list(self.forbidden_hits),
        }


def score_example(golden: ReviewExample, prediction: Prediction) -> ExampleScore:
    """Score one model prediction against one golden example."""

    exact = float(normalize_text(prediction.prediction) == normalize_text(golden.target_comment))
    lexical = token_f1(prediction.prediction, golden.target_comment)
    bleu = bleu_lite(prediction.prediction, golden.target_comment)
    rouge = rouge_l_lite(prediction.prediction, golden.target_comment)
    ratio = length_ratio(prediction.prediction, golden.target_comment)
    mention = phrase_recall(prediction.prediction, golden.must_mention)
    forbidden = forbidden_rate(prediction.prediction, golden.avoid)
    missed = missed_must_mention(prediction.prediction, golden.must_mention)
    hits = forbidden_hits(prediction.prediction, golden.avoid)
    predicted_severity = prediction.severity or infer_severity(prediction.prediction)
    severity_accuracy = float(predicted_severity == golden.severity)
    tags = f1_for_sets(prediction.tags, golden.tags)
    # composite keeps original weights for core signals; n-gram metrics folded into lexical share
    composite = (
        0.25 * lexical
        + 0.10 * bleu
        + 0.10 * rouge
        + 0.05 * ratio
        + 0.20 * mention
        + 0.15 * severity_accuracy
        + 0.10 * tags
        + 0.05 * (1.0 - forbidden)
    )
    return ExampleScore(
        id=golden.id,
        exact_match=exact,
        token_f1=lexical,
        bleu_lite=bleu,
        rouge_l=rouge,
        length_ratio=ratio,
        must_mention_recall=mention,
        forbidden_rate=forbidden,
        severity_accuracy=severity_accuracy,
        tag_f1=tags,
        composite=composite,
        language=golden.language,
        severity=golden.severity,
        predicted_severity=predicted_severity or "",
        missed_must_mention=missed,
        forbidden_hits=hits,
    )


def aggregate_scores(scores: list[ExampleScore]) -> dict[str, float | int | dict]:
    """Aggregate per-example scores into a report with breakdowns."""

    if not scores:
        return {"count": 0}
    fields = [
        "exact_match",
        "token_f1",
        "bleu_lite",
        "rouge_l",
        "length_ratio",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "composite",
    ]
    report: dict[str, float | int | dict] = {"count": len(scores)}
    for field_name in fields:
        values = [float(getattr(score, field_name)) for score in scores]
        report[field_name] = statistics.fmean(values)
        report[f"{field_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    report["composite_percent"] = math.floor(float(report["composite"]) * 10000) / 100
    ci_lo, ci_hi = bootstrap_ci([score.composite for score in scores])
    report["composite_ci_lo"] = ci_lo
    report["composite_ci_hi"] = ci_hi
    report["composite_ci"] = [ci_lo, ci_hi]

    by_language: dict[str, list[float]] = defaultdict(list)
    by_severity: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        by_language[score.language or "unknown"].append(score.composite)
        by_severity[score.severity or "unknown"].append(score.composite)
    report["by_language"] = {
        lang: round(statistics.fmean(vals), 6) for lang, vals in sorted(by_language.items())
    }
    report["by_severity"] = {
        sev: round(statistics.fmean(vals), 6) for sev, vals in sorted(by_severity.items())
    }
    return report


def error_analysis(
    scores: Sequence[ExampleScore],
    golden: Sequence[ReviewExample],
) -> dict[str, list[dict[str, str | int]]]:
    """Summarize missed rubric phrases, forbidden hits, and severity confusion."""

    golden_by_id = {example.id: example for example in golden}
    missed_counts: Counter[str] = Counter()
    forbidden_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for score in scores:
        example = golden_by_id.get(score.id)
        missed = score.missed_must_mention
        hits = score.forbidden_hits
        gold_severity = score.severity or (example.severity if example else "")
        pred_severity = score.predicted_severity
        for phrase in missed:
            missed_counts[phrase] += 1
        for phrase in hits:
            forbidden_counts[phrase] += 1
        if gold_severity:
            confusion_counts[(pred_severity or "unknown", gold_severity)] += 1

    return {
        "most_missed_must_mention": [
            {"phrase": phrase, "count": count} for phrase, count in missed_counts.most_common()
        ],
        "forbidden_phrase_hits": [
            {"phrase": phrase, "count": count} for phrase, count in forbidden_counts.most_common()
        ],
        "severity_confusion": [
            {"predicted": predicted, "gold": gold, "count": count}
            for (predicted, gold), count in sorted(
                confusion_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
        ],
    }


def _analysis_from_payload(analysis: dict | None) -> dict[str, list]:
    if not analysis:
        return {
            "most_missed_must_mention": [],
            "forbidden_phrase_hits": [],
            "severity_confusion": [],
        }
    return analysis


def render_markdown_report(
    aggregate: dict[str, float | int | dict],
    *,
    title: str = "Code review eval report",
    analysis: dict | None = None,
    per_example: list[dict] | None = None,
) -> str:
    """Render aggregate metrics as a simple Markdown document."""

    lines = [f"# {title}", "", f"**Examples:** {aggregate.get('count', 0)}", ""]
    if "composite_ci_lo" in aggregate and "composite_ci_hi" in aggregate:
        lines.append(
            f"**Composite 95% CI:** {float(aggregate['composite_ci_lo']):.4f} – "
            f"{float(aggregate['composite_ci_hi']):.4f}"
        )
        lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("| --- | ---: | ---: |")
    for key in (
        "composite",
        "token_f1",
        "bleu_lite",
        "rouge_l",
        "length_ratio",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "exact_match",
    ):
        if key not in aggregate:
            continue
        mean = float(aggregate[key])  # type: ignore[arg-type]
        std = float(aggregate.get(f"{key}_std", 0.0))  # type: ignore[arg-type]
        lines.append(f"| `{key}` | {mean:.4f} | {std:.4f} |")
    lines.append("")
    if isinstance(aggregate.get("by_language"), dict) and aggregate["by_language"]:
        lines.append("## By language")
        lines.append("")
        for lang, val in aggregate["by_language"].items():  # type: ignore[union-attr]
            lines.append(f"- **{lang}**: {float(val):.4f}")
        lines.append("")
    if isinstance(aggregate.get("by_severity"), dict) and aggregate["by_severity"]:
        lines.append("## By severity")
        lines.append("")
        for sev, val in aggregate["by_severity"].items():  # type: ignore[union-attr]
            lines.append(f"- **{sev}**: {float(val):.4f}")
        lines.append("")

    payload = _analysis_from_payload(analysis)
    lines.append("## Error analysis")
    lines.append("")
    lines.append("### Most-missed must_mention")
    lines.append("")
    missed = payload.get("most_missed_must_mention") or []
    if missed:
        lines.append("| Phrase | Count |")
        lines.append("| --- | ---: |")
        for row in missed:
            lines.append(f"| {row.get('phrase', '')} | {int(row.get('count', 0))} |")
    else:
        lines.append("No missed rubric phrases.")
    lines.append("")
    lines.append("### Forbidden-phrase hits")
    lines.append("")
    hits = payload.get("forbidden_phrase_hits") or []
    if hits:
        lines.append("| Phrase | Count |")
        lines.append("| --- | ---: |")
        for row in hits:
            lines.append(f"| {row.get('phrase', '')} | {int(row.get('count', 0))} |")
    else:
        lines.append("No forbidden-phrase hits.")
    lines.append("")
    lines.append("### Severity confusion")
    lines.append("")
    confusion = payload.get("severity_confusion") or []
    if confusion:
        lines.append("| Predicted | Gold | Count |")
        lines.append("| --- | --- | ---: |")
        for row in confusion:
            predicted = row.get("predicted", "")
            gold = row.get("gold", "")
            count = int(row.get("count", 0))
            lines.append(f"| {predicted} | {gold} | {count} |")
    else:
        lines.append("No severity pairs recorded.")
    lines.append("")

    if per_example:
        ranked = sorted(per_example, key=lambda row: float(row.get("composite", 0.0)))
        lines.append("## Examples (worst composite first)")
        lines.append("")
        lines.append("| ID | Language | Severity | Composite | Mention recall |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for row in ranked:
            lines.append(
                f"| `{row.get('id', '')}` | {row.get('language', '')} | "
                f"{row.get('severity', '')} | {float(row.get('composite', 0.0)):.4f} | "
                f"{float(row.get('must_mention_recall', 0.0)):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_html_report(
    aggregate: dict,
    per_example: list[dict],
    analysis: dict | None = None,
) -> str:
    """Render a self-contained HTML report (inline CSS, no CDN)."""

    payload = _analysis_from_payload(analysis)
    count = int(aggregate.get("count", len(per_example)))
    composite = float(aggregate.get("composite", 0.0) or 0.0)
    ci_lo = aggregate.get("composite_ci_lo")
    ci_hi = aggregate.get("composite_ci_hi")
    if ci_lo is None or ci_hi is None:
        ci = aggregate.get("composite_ci") or []
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_lo, ci_hi = ci
    ci_html = ""
    if ci_lo is not None and ci_hi is not None:
        ci_html = (
            f'<p class="ci">Composite 95% CI '
            f"<strong>{float(ci_lo):.4f} – {float(ci_hi):.4f}</strong></p>"
        )

    metric_keys = (
        "composite",
        "token_f1",
        "bleu_lite",
        "rouge_l",
        "length_ratio",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "exact_match",
    )
    metric_rows = []
    for key in metric_keys:
        if key not in aggregate:
            continue
        mean = float(aggregate[key])
        std = float(aggregate.get(f"{key}_std", 0.0) or 0.0)
        metric_rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td>"
            f"<td class='num'>{mean:.4f}</td><td class='num'>{std:.4f}</td></tr>"
        )

    def _kv_list(mapping: object) -> str:
        if not isinstance(mapping, dict) or not mapping:
            return "<p class='empty'>None</p>"
        items = "".join(
            f"<li><strong>{html.escape(str(name))}</strong> "
            f"{float(value):.4f}</li>"
            for name, value in mapping.items()
        )
        return f"<ul>{items}</ul>"

    def _phrase_table(rows: list, empty: str) -> str:
        if not rows:
            return f"<p class='empty'>{html.escape(empty)}</p>"
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('phrase', '')))}</td>"
            f"<td class='num'>{int(row.get('count', 0))}</td>"
            "</tr>"
            for row in rows
        )
        return (
            "<table><thead><tr><th>Phrase</th><th>Count</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    confusion = payload.get("severity_confusion") or []
    if confusion:
        confusion_body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('predicted', '')))}</td>"
            f"<td>{html.escape(str(row.get('gold', '')))}</td>"
            f"<td class='num'>{int(row.get('count', 0))}</td>"
            "</tr>"
            for row in confusion
        )
        confusion_html = (
            "<table><thead><tr><th>Predicted</th><th>Gold</th><th>Count</th></tr></thead>"
            f"<tbody>{confusion_body}</tbody></table>"
        )
    else:
        confusion_html = "<p class='empty'>No severity pairs recorded.</p>"

    ranked = sorted(per_example, key=lambda row: float(row.get("composite", 0.0)))
    example_rows = []
    for index, row in enumerate(ranked):
        worst = " worst" if index == 0 and ranked else ""
        example_rows.append(
            f"<tr class='{worst.strip()}'>"
            f"<td><code>{html.escape(str(row.get('id', '')))}</code></td>"
            f"<td>{html.escape(str(row.get('language', '')))}</td>"
            f"<td>{html.escape(str(row.get('severity', '')))}</td>"
            f"<td class='num'>{float(row.get('composite', 0.0)):.4f}</td>"
            f"<td class='num'>{float(row.get('token_f1', 0.0)):.4f}</td>"
            f"<td class='num'>{float(row.get('must_mention_recall', 0.0)):.4f}</td>"
            f"<td class='num'>{float(row.get('forbidden_rate', 0.0)):.4f}</td>"
            f"<td class='num'>{float(row.get('severity_accuracy', 0.0)):.4f}</td>"
            "</tr>"
        )
    examples_html = (
        "<table><thead><tr>"
        "<th>ID</th><th>Language</th><th>Severity</th><th>Composite</th>"
        "<th>Token F1</th><th>Mention recall</th><th>Forbidden</th><th>Severity acc.</th>"
        f"</tr></thead><tbody>{''.join(example_rows)}</tbody></table>"
        if example_rows
        else "<p class='empty'>No per-example scores.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Code review eval report</title>
<style>
:root {{
  --ink: #12202b;
  --muted: #5b6b75;
  --line: #d7e0e6;
  --paper: #f6f8f7;
  --card: #ffffff;
  --accent: #0f6f62;
  --worst: #fff4f1;
  --worst-edge: #d45a3a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font: 15px/1.5 "Segoe UI", system-ui, sans-serif;
}}
header {{
  background: var(--ink);
  color: #f4f7f6;
  padding: 1.6rem 1.4rem 1.3rem;
}}
header p {{ margin: 0.35rem 0 0; color: #c5d0d6; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 1.4rem; }}
h1, h2 {{ letter-spacing: -0.02em; }}
h1 {{ margin: 0; font-size: 1.7rem; }}
h2 {{ margin: 1.6rem 0 0.6rem; font-size: 1.15rem; }}
.score {{ font-size: 2rem; font-weight: 700; color: #8ee0d2; }}
.ci {{ color: #d5e6ea; }}
section {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1rem 1.1rem 1.15rem;
  margin-bottom: 1rem;
}}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 0.45rem 0.5rem; text-align: left; }}
th {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.worst td {{ background: var(--worst); }}
tr.worst td:first-child {{ box-shadow: inset 3px 0 0 var(--worst-edge); }}
.empty {{ color: var(--muted); }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }}
ul {{ margin: 0.2rem 0 0; padding-left: 1.2rem; }}
.grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
</style>
</head>
<body>
<header>
  <h1>Code review eval report</h1>
  <p>{count} example{'s' if count != 1 else ''}</p>
  <p class="score">composite {composite:.4f}</p>
  {ci_html}
</header>
<main>
<section>
  <h2>Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Mean</th><th>Std</th></tr></thead>
  <tbody>{''.join(metric_rows)}</tbody></table>
</section>
<div class="grid">
<section>
  <h2>By language</h2>
  {_kv_list(aggregate.get("by_language"))}
</section>
<section>
  <h2>By severity</h2>
  {_kv_list(aggregate.get("by_severity"))}
</section>
</div>
<section>
  <h2>Most-missed must_mention</h2>
  {_phrase_table(payload.get("most_missed_must_mention") or [], "No missed rubric phrases.")}
</section>
<section>
  <h2>Forbidden-phrase hits</h2>
  {_phrase_table(payload.get("forbidden_phrase_hits") or [], "No forbidden-phrase hits.")}
</section>
<section>
  <h2>Severity confusion</h2>
  {confusion_html}
</section>
<section>
  <h2>Examples by worst composite</h2>
  {examples_html}
</section>
</main>
</body>
</html>
"""
