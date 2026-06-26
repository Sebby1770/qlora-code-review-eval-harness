from review_tuner.metrics import (
    ScoreWeights,
    f1_for_sets,
    forbidden_rate,
    infer_severity,
    normalize_text,
    phrase_recall,
    score_example,
    token_f1,
)
from review_tuner.schema import Prediction, ReviewExample


def test_token_f1_rewards_overlap() -> None:
    assert token_f1("restore the expires_at check", "please restore expires_at validation") > 0.4
    assert token_f1("completely different", "please restore expires_at validation") == 0.0


def test_phrase_recall_counts_required_phrases() -> None:
    assert phrase_recall("Please restore the expires_at check.", ("expires_at check",)) == 1.0
    assert phrase_recall("Please restore validation.", ("expires_at check",)) == 0.0


def test_score_example_combines_rubric_and_metadata() -> None:
    golden = ReviewExample(
        id="golden",
        diff="- expired check\n+ no check",
        file_path="auth.py",
        language="python",
        context="Refresh token flow.",
        target_comment="Restore the expires_at check and add a regression test.",
        severity="high",
        tags=("security", "tests"),
        must_mention=("expires_at check", "regression test"),
    )
    prediction = Prediction(
        id="golden",
        prediction="Restore the expires_at check and add a regression test.",
        severity="high",
        tags=("security", "tests"),
    )

    score = score_example(golden, prediction)

    assert score.must_mention_recall == 1.0
    assert score.severity_accuracy == 1.0
    assert score.tag_f1 == 1.0
    assert score.composite > 0.9


def test_fused_score_path_matches_public_metric_contract() -> None:
    golden = ReviewExample(
        id="golden",
        diff="+ changed",
        file_path="service.py",
        language="python",
        context="Context.",
        target_comment="Check token expiry before updating state.",
        severity="high",
        tags=("security", "tests"),
        must_mention=("token expiry", "updating state"),
        avoid=("style only", "rename this"),
    )
    prediction = Prediction(
        id="golden",
        prediction="Severity high: check token expiry before updating state, not style only.",
        tags=("security",),
    )

    score = score_example(golden, prediction)

    assert score.exact_match == float(
        normalize_text(prediction.prediction) == normalize_text(golden.target_comment)
    )
    assert score.token_f1 == token_f1(prediction.prediction, golden.target_comment)
    assert score.must_mention_recall == phrase_recall(
        prediction.prediction, golden.must_mention
    )
    assert score.forbidden_rate == forbidden_rate(prediction.prediction, golden.avoid)
    assert score.severity_accuracy == float(
        infer_severity(prediction.prediction) == golden.severity
    )
    assert score.tag_f1 == f1_for_sets(prediction.tags, golden.tags)


def test_severity_inference_is_deterministic_when_multiple_labels_appear() -> None:
    assert infer_severity("[low] This is actually [blocker].") == "blocker"


def test_score_weights_are_validated_and_reusable() -> None:
    custom = ScoreWeights(
        token_f1=1.0,
        must_mention_recall=0.0,
        severity_accuracy=0.0,
        tag_f1=0.0,
        forbidden_absence=0.0,
    )
    golden = ReviewExample(
        id="golden",
        diff="+ changed",
        file_path="service.py",
        language="python",
        context="Context.",
        target_comment="exact target",
    )
    prediction = Prediction(id="golden", prediction="different target")

    assert score_example(golden, prediction, weights=custom).composite == token_f1(
        "different target",
        "exact target",
    )
