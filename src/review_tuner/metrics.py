"""Golden-set metrics for review comment generation."""

from __future__ import annotations

import math
import re
import statistics
import string
from collections import Counter, defaultdict
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

    def as_dict(self) -> dict[str, float | str]:
        return {
            "id": self.id,
            "language": self.language,
            "severity": self.severity,
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


def render_markdown_report(
    aggregate: dict[str, float | int | dict],
    *,
    title: str = "Code review eval report",
) -> str:
    """Render aggregate metrics as a simple Markdown document."""

    lines = [f"# {title}", "", f"**Examples:** {aggregate.get('count', 0)}", ""]
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
    return "\n".join(lines)
