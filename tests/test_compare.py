from review_tuner.compare import compare_predictions, format_compare, is_caught, main
from review_tuner.data import write_jsonl
from review_tuner.metrics import ExampleScore
from review_tuner.schema import Prediction, ReviewExample


def _example(example_id: str, mention: str) -> ReviewExample:
    return ReviewExample(
        id=example_id,
        diff="- old\n+ new",
        file_path="service.py",
        language="python",
        context="Context.",
        target_comment=f"Please mention {mention} in the review.",
        severity="high",
        tags=("security",),
        must_mention=(mention,),
    )


def test_compare_identifies_newly_caught_and_missed() -> None:
    golden = [_example("keep", "expires_at"), _example("flip", "overflow")]
    predictions_a = [
        Prediction(
            id="keep",
            prediction="Please mention expires_at in the review.",
            severity="high",
            tags=("security",),
        ),
        Prediction(
            id="flip",
            prediction="This looks fine.",
            severity="high",
            tags=("security",),
        ),
    ]
    predictions_b = [
        Prediction(
            id="keep",
            prediction="Unrelated comment that misses the phrase.",
            severity="high",
            tags=("security",),
        ),
        Prediction(
            id="flip",
            prediction="Please mention overflow in the review.",
            severity="high",
            tags=("security",),
        ),
    ]

    result = compare_predictions(golden, predictions_a, predictions_b)

    assert result.newly_caught == ("flip",)
    assert result.newly_missed == ("keep",)
    assert is_caught(
        ExampleScore(
            id="x",
            exact_match=0.0,
            token_f1=0.0,
            must_mention_recall=1.0,
            forbidden_rate=0.0,
            severity_accuracy=1.0,
            tag_f1=1.0,
            composite=1.0,
        )
    )
    assert not is_caught(
        ExampleScore(
            id="x",
            exact_match=0.0,
            token_f1=0.0,
            must_mention_recall=0.5,
            forbidden_rate=0.0,
            severity_accuracy=1.0,
            tag_f1=1.0,
            composite=0.4,
        )
    )
    assert result.per_example[0].caught_a is True
    assert result.per_example[0].caught_b is False
    assert result.per_example[1].caught_a is False
    assert result.per_example[1].caught_b is True
    text = format_compare(result)
    assert "newly caught" in text
    assert "flip" in text


def test_compare_cli_writes_json(tmp_path) -> None:
    golden_path = tmp_path / "golden.jsonl"
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    out_path = tmp_path / "compare.json"
    write_jsonl(
        golden_path,
        [
            {
                "id": "one",
                "language": "python",
                "file_path": "a.py",
                "context": "Context.",
                "diff": "- old\n+ new",
                "expected_comment": "Please mention overflow in the review.",
                "severity": "high",
                "tags": ["security"],
                "rubric": {"must_mention": ["overflow"]},
            }
        ],
    )
    write_jsonl(a_path, [{"id": "one", "prediction": "Looks fine.", "severity": "high"}])
    write_jsonl(
        b_path,
        [{"id": "one", "prediction": "Please mention overflow in the review.", "severity": "high"}],
    )

    exit_code = main(
        [
            "--golden",
            str(golden_path),
            "--a",
            str(a_path),
            "--b",
            str(b_path),
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    payload = out_path.read_text(encoding="utf-8")
    assert "newly_caught" in payload
    assert "one" in payload
