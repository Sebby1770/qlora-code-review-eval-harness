import pytest

from review_tuner.data import iter_examples, iter_jsonl, write_jsonl
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def valid_example(**overrides) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "one",
        "language": "python",
        "file_path": "service.py",
        "context": "A focused example.",
        "diff": "+ changed",
        "expected_comment": "Please verify the changed behavior.",
    }
    record.update(overrides)
    return record


def test_schema_rejects_strings_for_list_fields() -> None:
    with pytest.raises(DatasetError, match="tags must be a list"):
        ReviewExample.from_dict(valid_example(tags="security"))

    with pytest.raises(DatasetError, match="rubric.must_mention must be a list"):
        ReviewExample.from_dict(
            valid_example(rubric={"must_mention": "validation"})
        )


def test_schema_normalizes_and_deduplicates_list_fields() -> None:
    example = ReviewExample.from_dict(
        valid_example(tags=[" Security ", "security", "Tests"])
    )
    prediction = Prediction.from_dict(
        {"id": "one", "prediction": "Review comment", "tags": None}
    )

    assert example.tags == ("security", "tests")
    assert prediction.tags is None


def test_prediction_serialization_omits_unset_optional_fields() -> None:
    assert Prediction(id="one", prediction="Looks good.").to_record() == {
        "id": "one",
        "prediction": "Looks good.",
    }
    assert Prediction(
        id="one",
        prediction="Needs tests.",
        severity="medium",
        tags=("tests",),
    ).to_record() == {
        "id": "one",
        "prediction": "Needs tests.",
        "severity": "medium",
        "tags": ["tests"],
    }
    assert Prediction(
        id="one",
        prediction="No issue categories apply.",
        severity="low",
        tags=(),
    ).to_record() == {
        "id": "one",
        "prediction": "No issue categories apply.",
        "severity": "low",
        "tags": [],
    }


def test_schema_rejects_conflicting_target_fields() -> None:
    with pytest.raises(DatasetError, match="must not conflict"):
        ReviewExample.from_dict(
            valid_example(review_comment="A different target comment.")
        )


def test_jsonl_iteration_is_lazy_and_reports_source_location(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id": "one"}\nnot-json\n', encoding="utf-8")
    records = iter_jsonl(path)

    assert next(records) == {"id": "one"}
    with pytest.raises(DatasetError, match=r"examples\.jsonl:2: invalid JSON"):
        next(records)


def test_iter_examples_adds_path_to_schema_errors(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [valid_example(tags="security")])

    with pytest.raises(DatasetError, match=r"examples\.jsonl:row 1"):
        next(iter_examples(path))


def test_atomic_write_preserves_existing_file_on_stream_failure(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text("existing\n", encoding="utf-8")

    def failing_records():
        yield {"id": "one", "prediction": "ok"}
        raise DatasetError("generation failed")

    with pytest.raises(DatasetError, match="generation failed"):
        write_jsonl(path, failing_records())

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".predictions.jsonl.*.tmp"))


def test_atomic_write_cleans_up_unexpected_generation_failures(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"

    def failing_records():
        yield {"id": "one", "prediction": "ok"}
        raise RuntimeError("model generation failed")

    with pytest.raises(RuntimeError, match="model generation failed"):
        write_jsonl(path, failing_records())

    assert not path.exists()
    assert not list(tmp_path.glob(".predictions.jsonl.*.tmp"))
