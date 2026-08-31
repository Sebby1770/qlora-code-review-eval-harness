from review_tuner.evaluate import heuristic_prediction
from review_tuner.schema import Prediction, ReviewExample
from review_tuner.view import build_eval_view, letter_grade, verdict


def _example(**kwargs) -> ReviewExample:
    base = dict(
        id="x",
        diff="- secret\n+ token",
        file_path="a.py",
        language="python",
        context="auth must reject expired tokens",
        target_comment="Restore expired refresh tokens and the expires_at check.",
        severity="high",
        tags=("security",),
        must_mention=("expired refresh tokens", "expires_at"),
        avoid=("style",),
    )
    base.update(kwargs)
    return ReviewExample(**base)


def test_histogram_and_tag_breakdown_and_overlap():
    from review_tuner.data import load_examples
    from review_tuner.evaluate import heuristic_prediction
    from review_tuner.view import build_eval_view, overlapping_tokens

    shared = overlapping_tokens(
        "restore expired tokens please",
        "expired tokens must be restored",
    )
    assert shared == ["expired", "tokens"]
    golden = load_examples("data/golden/code_review_golden.jsonl")
    view = build_eval_view(golden, [heuristic_prediction(example) for example in golden])
    hist = view["histogram"]
    assert sum(hist.values()) == 15
    assert "security" in view["by_tag"]
    assert view["examples"][0]["overlap_tokens"]


def test_summarize_dataset_counts_languages_and_tags():
    from review_tuner.view import summarize_dataset

    summary = summarize_dataset(
        [
            {"language": "Python", "severity": "high", "tags": ["security", "tests"]},
            {"language": "python", "severity": "low", "tags": ["tests"]},
            {"language": "go"},
        ]
    )
    assert summary["count"] == 3
    assert summary["languages"]["python"] == 2
    assert summary["languages"]["go"] == 1
    assert summary["tags"]["tests"] == 2
    assert summary["severities"]["(missing)"] == 1


def test_letter_grade_boundaries():
    assert letter_grade(0.80) == "A"
    assert letter_grade(0.79) == "B"
    assert letter_grade(0.65) == "B"
    assert letter_grade(0.50) == "C"
    assert letter_grade(0.35) == "D"
    assert letter_grade(0.34) == "F"
    assert letter_grade(1.2) == "A"
    assert letter_grade(-0.1) == "F"


def test_verdict_gate():
    assert verdict(0.60, 0.60) == "Pass"
    assert verdict(0.50, 0.60) == "Weak"
    assert verdict(0.40, 0.60) == "Fail"


def test_story_mentions_missed_phrase():
    golden = [_example()]
    prediction = Prediction(id="x", prediction="Looks fine, maybe style tweaks.", severity="low")
    view = build_eval_view(golden, [prediction])
    assert "expired refresh tokens" in view["story"]
    assert view["examples"][0]["diff"]
    assert view["examples"][0]["explanation"]
    assert view["worst_id"] == "x"


def test_eval_view_sample_smoke():
    from review_tuner.data import load_examples, load_predictions

    golden = load_examples("data/golden/code_review_golden.jsonl")
    predictions = load_predictions("examples/predictions.sample.jsonl")
    view = build_eval_view(golden, predictions)
    assert view["aggregate"]["count"] == 15
    assert abs(float(view["aggregate"]["composite"]) - 0.5438) < 0.02
    assert view["letter_grade"] in {"A", "B", "C", "D", "F"}
    assert view["examples"][0]["expected_comment"]


def test_baseline_view_runs():
    from review_tuner.data import load_examples

    golden = load_examples("data/golden/code_review_golden.jsonl")
    predictions = [heuristic_prediction(example) for example in golden]
    view = build_eval_view(golden, predictions)
    assert view["aggregate"]["count"] == 15
    assert "Overall" in view["story"]
