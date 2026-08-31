import statistics

from review_tuner.metrics import (
    aggregate_scores,
    bleu_lite,
    bootstrap_ci,
    error_analysis,
    infer_severity,
    length_ratio,
    render_html_report,
    rouge_l_lite,
    score_example,
    token_f1,
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


def test_infer_severity_uses_stable_order():
    assert infer_severity("Please treat this as blocker severity, not a nit") == "blocker"
    assert infer_severity("[high] missing authz") == "high"
    assert infer_severity("nit severity only") == "nit"


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
    assert "composite_ci" in agg
    assert agg["composite_ci_lo"] <= score.composite <= agg["composite_ci_hi"]


def test_bootstrap_ci_seeded_contains_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    lo, hi = bootstrap_ci(values, n=500, seed=0)
    mean = statistics.fmean(values)
    assert lo <= mean <= hi
    assert bootstrap_ci(values, n=500, seed=0) == (lo, hi)
    spread = [0.0, 1.0] * 20
    wide_lo, wide_hi = bootstrap_ci(spread, n=500, seed=0)
    assert wide_lo < wide_hi
    assert wide_lo <= statistics.fmean(spread) <= wide_hi


def test_error_analysis_counts_misses_hits_and_confusion():
    golden = _ex(
        must_mention=("expired tokens", "test"),
        avoid=("style",),
        severity="high",
    )
    pred = Prediction(
        id="x",
        prediction="please change the style of this function",
        severity="low",
        tags=(),
    )
    score = score_example(golden, pred)
    analysis = error_analysis([score], [golden])
    missed = {row["phrase"]: row["count"] for row in analysis["most_missed_must_mention"]}
    assert missed["expired tokens"] == 1
    assert missed["test"] == 1
    hits = {row["phrase"]: row["count"] for row in analysis["forbidden_phrase_hits"]}
    assert hits["style"] == 1
    assert any(
        row["predicted"] == "low" and row["gold"] == "high"
        for row in analysis["severity_confusion"]
    )


def test_html_report_contains_composite_and_example_id():
    golden = _ex()
    pred = Prediction(
        id="x",
        prediction="Please restore expired tokens validation and add a test.",
        severity="high",
        tags=("security", "tests"),
    )
    score = score_example(golden, pred)
    aggregate = aggregate_scores([score])
    analysis = error_analysis([score], [golden])
    html = render_html_report(aggregate, [score.as_dict()], analysis)
    assert "composite" in html.lower()
    assert "x" in html
