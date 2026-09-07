from review_tuner.metrics import (
    ScoreWeights,
    bleu_lite,
    extend_score,
    f1_for_sets,
    forbidden_rate,
    infer_severity,
    length_ratio,
    missed_phrases,
    normalize_text,
    phrase_recall,
    rouge_l_lite,
    score_example,
    security_fail,
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


def test_example_score_as_dict_keys_stay_stable() -> None:
    golden = ReviewExample(
        id="golden",
        diff="+ changed",
        file_path="service.py",
        language="python",
        context="Context.",
        target_comment="Check token expiry before updating state.",
        severity="high",
        tags=("security",),
        must_mention=("token expiry",),
    )
    prediction = Prediction(id="golden", prediction="Looks fine to me.")
    score = score_example(golden, prediction)
    extended = extend_score(golden, prediction, score)

    assert set(score.as_dict()) == {
        "id",
        "exact_match",
        "token_f1",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "composite",
    }
    assert extended.as_dict() == score.as_dict()
    assert extended.security_fail == 1.0
    assert extended.missed_must_mention == ("token expiry",)


def test_additive_lexical_metrics_and_security_fail() -> None:
    assert bleu_lite("the cat sat", "the cat sat") == 1.0
    assert rouge_l_lite("the cat sat", "the cat sat") == 1.0
    assert length_ratio("the cat sat", "the cat sat") == 1.0
    assert bleu_lite("", "hello") == 0.0
    assert rouge_l_lite("", "hello") == 0.0
    assert security_fail("high", 0.5) == 1.0
    assert security_fail("high", 1.0) == 0.0
    assert security_fail("low", 0.0) == 0.0
    assert missed_phrases("restore validation", ("expires_at check", "validation")) == (
        "expires_at check",
    )
