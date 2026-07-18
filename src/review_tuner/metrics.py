"""Golden-set metrics for review comment generation."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappush, heapreplace

from review_tuner.schema import SEVERITY_ORDER, Prediction, ReviewExample

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
    for severity in SEVERITY_ORDER:
        if f"[{severity}]" in text.lower() or f"{severity} severity" in normalized:
            return severity
    return None


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Composite score policy with validation in one place."""

    token_f1: float = 0.35
    must_mention_recall: float = 0.25
    severity_accuracy: float = 0.20
    tag_f1: float = 0.15
    forbidden_absence: float = 0.05

    def __post_init__(self) -> None:
        weights = (
            self.token_f1,
            self.must_mention_recall,
            self.severity_accuracy,
            self.tag_f1,
            self.forbidden_absence,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("score weights must be non-negative")
        if not math.isclose(sum(weights), 1.0):
            raise ValueError("score weights must sum to 1.0")


DEFAULT_SCORE_WEIGHTS = ScoreWeights()


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

    def metrics(self) -> dict[str, float]:
        """Return metric values without the example identifier."""

        return {
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "must_mention_recall": self.must_mention_recall,
            "forbidden_rate": self.forbidden_rate,
            "severity_accuracy": self.severity_accuracy,
            "tag_f1": self.tag_f1,
            "composite": self.composite,
        }

    def values(self) -> tuple[float, ...]:
        """Return metric values in the canonical aggregation order."""

        return (
            self.exact_match,
            self.token_f1,
            self.must_mention_recall,
            self.forbidden_rate,
            self.severity_accuracy,
            self.tag_f1,
            self.composite,
        )

    def as_dict(self) -> dict[str, float | str]:
        return {"id": self.id, **self.metrics()}


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
        for statistic, value in zip(
            self._statistics.values(), score.values(), strict=True
        ):
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


_OUTCOME_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScoreComparison:
    """Paired candidate and baseline scores for one golden example."""

    candidate: ExampleScore
    baseline: ExampleScore

    def __post_init__(self) -> None:
        if self.candidate.id != self.baseline.id:
            raise ValueError("candidate and baseline score ids must match")

    @property
    def deltas(self) -> tuple[float, ...]:
        """Return candidate-minus-baseline deltas in canonical metric order."""

        return tuple(
            candidate - baseline
            for candidate, baseline in zip(
                self.candidate.values(), self.baseline.values(), strict=True
            )
        )

    @property
    def outcome(self) -> str:
        """Classify the paired result using the composite metric."""

        delta = self.candidate.composite - self.baseline.composite
        if math.isclose(delta, 0.0, abs_tol=_OUTCOME_TOLERANCE):
            return "tie"
        return "win" if delta > 0.0 else "loss"

    def as_dict(self) -> dict[str, object]:
        """Serialize without changing the existing top-level candidate fields."""

        record: dict[str, object] = dict(self.candidate.as_dict())
        record["baseline"] = self.baseline.metrics()
        record["delta"] = dict(zip(_SCORE_FIELDS, self.deltas, strict=True))
        record["outcome"] = self.outcome
        return record


class ScoreComparisonAccumulator:
    """Aggregate paired scores and retain only the largest bounded regressions."""

    def __init__(self, *, top_regressions: int = 10) -> None:
        if top_regressions < 0:
            raise ValueError("top_regressions must be non-negative")
        self._candidate = ScoreAccumulator()
        self._baseline = ScoreAccumulator()
        self._deltas = {name: _RunningStatistic() for name in _SCORE_FIELDS}
        self._wins = 0
        self._ties = 0
        self._losses = 0
        self._top_regressions = top_regressions
        self._regressions: list[tuple[float, str, float, float]] = []

    @property
    def count(self) -> int:
        return self._candidate.count

    def add(self, comparison: ScoreComparison) -> None:
        self._candidate.add(comparison.candidate)
        self._baseline.add(comparison.baseline)
        for statistic, delta in zip(
            self._deltas.values(), comparison.deltas, strict=True
        ):
            statistic.add(delta)

        if comparison.outcome == "win":
            self._wins += 1
        elif comparison.outcome == "tie":
            self._ties += 1
        else:
            self._losses += 1
            self._retain_regression(comparison)

    def _retain_regression(self, comparison: ScoreComparison) -> None:
        if self._top_regressions == 0:
            return
        item = (
            comparison.baseline.composite - comparison.candidate.composite,
            comparison.candidate.id,
            comparison.candidate.composite,
            comparison.baseline.composite,
        )
        if len(self._regressions) < self._top_regressions:
            heappush(self._regressions, item)
        elif item > self._regressions[0]:
            heapreplace(self._regressions, item)

    def report(self) -> dict[str, object]:
        """Return candidate, baseline, paired deltas, outcomes, and regressions."""

        count = self.count
        delta_report: dict[str, float | int] = {"count": count}
        for name, statistic in self._deltas.items():
            delta_report[name] = statistic.mean
            delta_report[f"{name}_std"] = statistic.population_stddev
        delta_report["composite_percent_points"] = (
            float(delta_report["composite"]) * 100.0 if count else 0.0
        )

        outcomes: dict[str, float | int] = {
            "candidate_wins": self._wins,
            "ties": self._ties,
            "candidate_losses": self._losses,
            "win_rate": self._wins / count if count else 0.0,
            "non_regression_rate": (self._wins + self._ties) / count if count else 0.0,
        }
        regressions = [
            {
                "id": example_id,
                "candidate_composite": candidate,
                "baseline_composite": baseline,
                "composite_delta": candidate - baseline,
            }
            for _, example_id, candidate, baseline in sorted(
                self._regressions, reverse=True
            )
        ]
        return {
            "candidate": self._candidate.report(),
            "baseline": self._baseline.report(),
            "delta": delta_report,
            "outcomes": outcomes,
            "largest_regressions": regressions,
        }


def score_example(
    golden: ReviewExample,
    prediction: Prediction,
    *,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> ExampleScore:
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
    severity_accuracy = float(
        prediction.severity is not None and prediction.severity == golden.severity
    )
    tags = (
        f1_for_sets(prediction.tags, golden.tags)
        if prediction.tags is not None
        else 0.0
    )
    composite = (
        weights.token_f1 * lexical
        + weights.must_mention_recall * mention
        + weights.severity_accuracy * severity_accuracy
        + weights.tag_f1 * tags
        + weights.forbidden_absence * (1.0 - forbidden)
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
