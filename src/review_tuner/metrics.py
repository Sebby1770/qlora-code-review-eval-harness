"""Golden-set metrics for review comment generation."""

from __future__ import annotations

import math
import re
import statistics
import string
from collections import Counter
from dataclasses import dataclass

from review_tuner.schema import VALID_SEVERITIES, Prediction, ReviewExample

_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_text(value: str) -> str:
    """Lowercase and collapse punctuation and whitespace."""

    return " ".join(value.lower().translate(_PUNCT_TRANSLATION).split())


def tokenize(value: str) -> list[str]:
    """Tokenize text for lightweight lexical metrics."""

    return normalize_text(value).split()


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
    must_mention_recall: float
    forbidden_rate: float
    severity_accuracy: float
    tag_f1: float
    composite: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "id": self.id,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
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
    mention = phrase_recall(prediction.prediction, golden.must_mention)
    forbidden = forbidden_rate(prediction.prediction, golden.avoid)
    predicted_severity = prediction.severity or infer_severity(prediction.prediction)
    severity_accuracy = float(predicted_severity == golden.severity)
    tags = f1_for_sets(prediction.tags, golden.tags)
    composite = (
        0.35 * lexical
        + 0.25 * mention
        + 0.20 * severity_accuracy
        + 0.15 * tags
        + 0.05 * (1.0 - forbidden)
    )
    return ExampleScore(
        id=golden.id,
        exact_match=exact,
        token_f1=lexical,
        must_mention_recall=mention,
        forbidden_rate=forbidden,
        severity_accuracy=severity_accuracy,
        tag_f1=tags,
        composite=composite,
    )


def aggregate_scores(scores: list[ExampleScore]) -> dict[str, float | int]:
    """Aggregate per-example scores into a report."""

    if not scores:
        return {"count": 0}
    fields = [
        "exact_match",
        "token_f1",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "composite",
    ]
    report: dict[str, float | int] = {"count": len(scores)}
    for field_name in fields:
        values = [float(getattr(score, field_name)) for score in scores]
        report[field_name] = statistics.fmean(values)
        report[f"{field_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    report["composite_percent"] = math.floor(float(report["composite"]) * 10000) / 100
    return report
