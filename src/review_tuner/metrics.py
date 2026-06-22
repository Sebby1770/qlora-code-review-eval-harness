"""Golden-set metrics for review comment generation."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable
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

    return _token_f1_from_tokens(tokenize(prediction), tokenize(target))


def _token_f1_from_tokens(pred_tokens: list[str], target_tokens: list[str]) -> float:
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
    return _phrase_rate(normalize_text(prediction), required_phrases)


def forbidden_rate(prediction: str, forbidden_phrases: tuple[str, ...]) -> float:
    """Score the share of forbidden phrases that appear in the prediction."""

    if not forbidden_phrases:
        return 0.0
    return _phrase_rate(normalize_text(prediction), forbidden_phrases)


def _phrase_rate(normalized_prediction: str, phrases: tuple[str, ...]) -> float:
    hits = sum(1 for phrase in phrases if normalize_text(phrase) in normalized_prediction)
    return hits / len(phrases)


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
    return _infer_severity_from_normalized(text, normalized)


def _infer_severity_from_normalized(text: str, normalized: str) -> str | None:
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


_SCORE_FIELDS = (
    "exact_match",
    "token_f1",
    "must_mention_recall",
    "forbidden_rate",
    "severity_accuracy",
    "tag_f1",
    "composite",
)


@dataclass(slots=True)
class _RunningStatistic:
    count: int = 0
    mean: float = 0.0
    squared_distance: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_distance += delta * (value - self.mean)

    @property
    def population_stddev(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.squared_distance / self.count)


class ScoreAccumulator:
    """Compute score means and population deviations in constant memory."""

    def __init__(self) -> None:
        self._statistics = {name: _RunningStatistic() for name in _SCORE_FIELDS}

    @property
    def count(self) -> int:
        return self._statistics["composite"].count

    def add(self, score: ExampleScore) -> None:
        values = (
            score.exact_match,
            score.token_f1,
            score.must_mention_recall,
            score.forbidden_rate,
            score.severity_accuracy,
            score.tag_f1,
            score.composite,
        )
        for statistic, value in zip(self._statistics.values(), values, strict=True):
            statistic.add(value)

    def report(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        report: dict[str, float | int] = {"count": self.count}
        for name, statistic in self._statistics.items():
            report[name] = statistic.mean
            report[f"{name}_std"] = statistic.population_stddev
        report["composite_percent"] = math.floor(
            float(report["composite"]) * 10000
        ) / 100
        return report


def score_example(golden: ReviewExample, prediction: Prediction) -> ExampleScore:
    """Score one model prediction against one golden example."""

    normalized_prediction = normalize_text(prediction.prediction)
    normalized_target = normalize_text(golden.target_comment)
    exact = float(normalized_prediction == normalized_target)
    lexical = _token_f1_from_tokens(
        normalized_prediction.split(),
        normalized_target.split(),
    )
    mention = (
        _phrase_rate(normalized_prediction, golden.must_mention)
        if golden.must_mention
        else 1.0
    )
    forbidden = (
        _phrase_rate(normalized_prediction, golden.avoid) if golden.avoid else 0.0
    )
    predicted_severity = prediction.severity or _infer_severity_from_normalized(
        prediction.prediction,
        normalized_prediction,
    )
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


def aggregate_scores(scores: Iterable[ExampleScore]) -> dict[str, float | int]:
    """Aggregate per-example scores in one pass and constant memory."""

    accumulator = ScoreAccumulator()
    for score in scores:
        accumulator.add(score)
    return accumulator.report()
