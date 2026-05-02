from review_tuner.metrics import phrase_recall, score_example, token_f1
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
