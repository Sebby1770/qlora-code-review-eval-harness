from review_tuner.metrics import (
    bleu_lite,
    length_ratio,
    rouge_l_lite,
    token_f1,
    aggregate_scores,
    score_example,
)
from review_tuner.schema import Prediction, ReviewExample


def _ex(**kwargs):
    base = dict(
        id="x",
        diff="diff",
        file_path="a.py",
        language="python",
        context="ctx",
        target_comment="please add a test for expired tokens",
        severity="high",
        tags=("security", "tests"),
        must_mention=("expired tokens", "test"),
        avoid=("style",),
    )
    base.update(kwargs)
    return ReviewExample(**base)


def test_token_f1_identical():
    assert token_f1("hello world", "hello world") == 1.0


def test_bleu_and_rouge_positive():
    pred = "please restore the expires_at check and add a regression test"
    ref = "please restore the expires_at check and add a regression test for expired tokens"
    assert bleu_lite(pred, ref) > 0.3
    assert rouge_l_lite(pred, ref) > 0.3
    assert 0.0 < length_ratio(pred, ref) <= 1.0


def test_score_example_composite_bounds():
    golden = _ex()
    pred = Prediction(
        id="x",
        prediction="Please restore expired tokens validation and add a test.",
        severity="high",
        tags=("security", "tests"),
    )
    score = score_example(golden, pred)
    assert 0.0 <= score.composite <= 1.0
    assert score.bleu_lite >= 0.0
    assert score.rouge_l >= 0.0
    agg = aggregate_scores([score])
    assert agg["count"] == 1
    assert "by_language" in agg
    assert "python" in agg["by_language"]
