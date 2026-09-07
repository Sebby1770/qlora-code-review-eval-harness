"""Compare two prediction JSONL files against the same golden set."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from review_tuner.data import index_by_id, iter_examples, iter_predictions, write_json
from review_tuner.evaluate import iter_scores
from review_tuner.metrics import ExampleScore, aggregate_scores
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def is_caught(score: ExampleScore) -> bool:
    """A case is caught when every required rubric phrase is mentioned."""

    return score.must_mention_recall >= 1.0


@dataclass(frozen=True)
class ExampleDelta:
    """Per-example comparison against two prediction files."""

    id: str
    composite_a: float
    composite_b: float
    delta_composite: float
    caught_a: bool
    caught_b: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "id": self.id,
            "composite_a": self.composite_a,
            "composite_b": self.composite_b,
            "delta_composite": self.delta_composite,
            "caught_a": self.caught_a,
            "caught_b": self.caught_b,
        }


@dataclass(frozen=True)
class CompareResult:
    """Summary of prediction file B relative to A."""

    delta_composite: float
    newly_caught: tuple[str, ...]
    newly_missed: tuple[str, ...]
    per_example: tuple[ExampleDelta, ...]
    a: dict[str, float | int]
    b: dict[str, float | int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_composite": self.delta_composite,
            "newly_caught": list(self.newly_caught),
            "newly_missed": list(self.newly_missed),
            "per_example": [row.as_dict() for row in self.per_example],
            "a": self.a,
            "b": self.b,
        }


def compare_predictions(
    golden: Sequence[ReviewExample],
    predictions_a: Sequence[Prediction],
    predictions_b: Sequence[Prediction],
) -> CompareResult:
    """Score two aligned prediction files and diff caught/missed cases."""

    scores_a = list(iter_scores(golden, predictions_a))
    scores_b = list(iter_scores(golden, predictions_b))
    by_b = index_by_id(scores_b)
    rows: list[ExampleDelta] = []
    newly_caught: list[str] = []
    newly_missed: list[str] = []
    for score_a in scores_a:
        score_b = by_b[score_a.id]
        caught_a = is_caught(score_a)
        caught_b = is_caught(score_b)
        rows.append(
            ExampleDelta(
                id=score_a.id,
                composite_a=score_a.composite,
                composite_b=score_b.composite,
                delta_composite=score_b.composite - score_a.composite,
                caught_a=caught_a,
                caught_b=caught_b,
            )
        )
        if caught_b and not caught_a:
            newly_caught.append(score_a.id)
        elif caught_a and not caught_b:
            newly_missed.append(score_a.id)
    aggregate_a = aggregate_scores(scores_a)
    aggregate_b = aggregate_scores(scores_b)
    return CompareResult(
        delta_composite=float(aggregate_b.get("composite", 0.0))
        - float(aggregate_a.get("composite", 0.0)),
        newly_caught=tuple(newly_caught),
        newly_missed=tuple(newly_missed),
        per_example=tuple(rows),
        a=aggregate_a,
        b=aggregate_b,
    )


def format_compare(result: CompareResult) -> str:
    """Human-readable comparison for the terminal."""

    def _ids(values: Sequence[str]) -> str:
        return ", ".join(values) if values else "(none)"

    lines = [
        f"A composite: {float(result.a.get('composite', 0.0)):.4f} (n={result.a.get('count', 0)})",
        f"B composite: {float(result.b.get('composite', 0.0)):.4f} (n={result.b.get('count', 0)})",
        f"delta composite: {result.delta_composite:+.4f}",
        f"newly caught ({len(result.newly_caught)}): {_ids(result.newly_caught)}",
        f"newly missed ({len(result.newly_missed)}): {_ids(result.newly_missed)}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="review-eval compare",
        description="Compare two prediction JSONL files against the same golden set.",
    )
    parser.add_argument("--golden", required=True, help="Path to golden JSONL examples.")
    parser.add_argument("--a", required=True, help="Baseline prediction JSONL.")
    parser.add_argument("--b", required=True, help="Candidate prediction JSONL.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args(argv)
    try:
        golden = list(iter_examples(args.golden))
        predictions_a = list(iter_predictions(args.a))
        predictions_b = list(iter_predictions(args.b))
        result = compare_predictions(golden, predictions_a, predictions_b)
        print(format_compare(result), end="")
        if args.out:
            write_json(args.out, result.as_dict())
        return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
