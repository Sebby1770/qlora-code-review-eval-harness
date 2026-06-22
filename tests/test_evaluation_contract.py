import pytest

from review_tuner.data import write_jsonl
from review_tuner.evaluate import evaluate_predictions, main
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def example(example_id: str) -> ReviewExample:
    return ReviewExample(
        id=example_id,
        diff="+ changed",
        file_path="service.py",
        language="python",
        context="A focused example.",
        target_comment="Please verify the changed behavior.",
    )


def prediction(prediction_id: str) -> Prediction:
    return Prediction(
        id=prediction_id,
        prediction="Please verify the changed behavior.",
    )


def test_evaluation_requires_exact_id_alignment() -> None:
    with pytest.raises(DatasetError, match="missing predictions.*unexpected predictions"):
        evaluate_predictions([example("golden")], [prediction("stale")])


def test_evaluation_rejects_duplicate_and_empty_golden_sets() -> None:
    with pytest.raises(DatasetError, match="duplicate golden id"):
        evaluate_predictions(
            [example("one"), example("one")],
            [prediction("one")],
        )

    with pytest.raises(DatasetError, match="at least one example"):
        evaluate_predictions([], [])


def test_alignment_failure_does_not_replace_existing_reports(tmp_path) -> None:
    golden_path = tmp_path / "golden.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    aggregate_path = tmp_path / "aggregate.json"
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text("existing report\n", encoding="utf-8")
    write_jsonl(
        golden_path,
        [
            {
                "id": "golden",
                "diff": "+ changed",
                "file_path": "service.py",
                "language": "python",
                "context": "Context.",
                "expected_comment": "Review this.",
            }
        ],
    )
    write_jsonl(
        predictions_path,
        [{"id": "stale", "prediction": "Stale prediction."}],
    )

    exit_code = main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(predictions_path),
            "--out",
            str(aggregate_path),
            "--per-example-out",
            str(examples_path),
        ]
    )

    assert exit_code == 1
    assert examples_path.read_text(encoding="utf-8") == "existing report\n"
    assert not aggregate_path.exists()
