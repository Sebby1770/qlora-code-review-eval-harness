"""Repeatable synthetic throughput benchmark for the evaluation harness."""

from __future__ import annotations

import argparse
import json
import statistics
import timeit
from dataclasses import asdict, dataclass

from review_tuner.evaluate import evaluate_predictions
from review_tuner.schema import Prediction, ReviewExample


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    examples: int
    repeat: int
    median_seconds: float
    examples_per_second: float


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def make_dataset(count: int) -> tuple[list[ReviewExample], list[Prediction]]:
    golden = [
        ReviewExample(
            id=str(index),
            language="python",
            file_path="service.py",
            context="Validate the changed behavior.",
            diff="+ changed",
            target_comment="Check the changed behavior and add a regression test.",
            severity="medium",
            tags=("tests",),
            must_mention=("changed behavior", "regression test"),
            avoid=("style only",),
        )
        for index in range(count)
    ]
    predictions = [
        Prediction(
            id=str(index),
            prediction=(
                "Severity medium: check the changed behavior and add a regression test."
            ),
            tags=("tests",),
        )
        for index in range(count)
    ]
    return golden, predictions


def benchmark(*, examples: int = 20_000, repeat: int = 7) -> BenchmarkResult:
    golden, predictions = make_dataset(examples)
    evaluate_predictions(golden, predictions)
    samples = timeit.repeat(
        lambda: evaluate_predictions(golden, predictions),
        repeat=repeat,
        number=1,
    )
    median_seconds = statistics.median(samples)
    return BenchmarkResult(
        examples=examples,
        repeat=repeat,
        median_seconds=median_seconds,
        examples_per_second=examples / median_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=positive_integer, default=20_000)
    parser.add_argument("--repeat", type=positive_integer, default=7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = benchmark(examples=args.examples, repeat=args.repeat)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(f"Median of {result.repeat} runs ({result.examples:,} examples each)")
        print(f"Evaluation: {result.median_seconds:.4f}s")
        print(f"Throughput: {result.examples_per_second:,.0f} examples/second")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
