from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from review_tuner.evaluate import heuristic_prediction
from review_tuner.metrics import DEFAULT_SCORE_WEIGHTS, score_example
from review_tuner.schema import Prediction, ReviewExample

ENGINE = (
    Path(__file__).resolve().parents[1] / "src" / "review_tuner" / "studio_web" / "eval-engine.js"
)


def test_composite_weights_are_locked_in_python_and_js() -> None:
    weights = DEFAULT_SCORE_WEIGHTS
    assert (
        weights.token_f1,
        weights.must_mention_recall,
        weights.severity_accuracy,
        weights.tag_f1,
        weights.forbidden_absence,
    ) == (0.35, 0.25, 0.20, 0.15, 0.05)
    source = ENGINE.read_text(encoding="utf-8")
    assert "tokenF1: 0.35" in source
    assert "mustMentionRecall: 0.25" in source
    assert "severityAccuracy: 0.2" in source
    assert "tagF1: 0.15" in source
    assert "forbiddenAbsence: 0.05" in source
    assert "function bootstrapCi" in source
    assert "function severityConfusion" in source


def test_python_composite_matches_weight_formula() -> None:
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
        avoid=("style only",),
    )
    prediction = Prediction(
        id="golden",
        prediction="Severity high: check token expiry before updating state, not style only.",
        tags=("security",),
    )
    score = score_example(golden, prediction)
    expected = (
        0.35 * score.token_f1
        + 0.25 * score.must_mention_recall
        + 0.20 * score.severity_accuracy
        + 0.15 * score.tag_f1
        + 0.05 * (1.0 - score.forbidden_rate)
    )
    assert score.composite == pytest.approx(expected)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_eval_engine_matches_python_on_fixture(tmp_path) -> None:
    golden = ReviewExample(
        id="golden",
        diff="- expires_at check\n+ no check",
        file_path="auth.py",
        language="python",
        context="Refresh token flow.",
        target_comment="Restore the expires_at check and add a regression test.",
        severity="high",
        tags=("security", "tests"),
        must_mention=("expires_at check", "regression test"),
        avoid=("style",),
    )
    prediction = heuristic_prediction(golden)
    python_score = score_example(golden, prediction)
    raw_golden = {
        "id": golden.id,
        "language": golden.language,
        "file_path": golden.file_path,
        "context": golden.context,
        "diff": golden.diff,
        "expected_comment": golden.target_comment,
        "severity": golden.severity,
        "tags": list(golden.tags),
        "rubric": {"must_mention": list(golden.must_mention), "avoid": list(golden.avoid)},
    }
    raw_prediction = prediction.to_record()
    script = tmp_path / "parity.js"
    script.write_text(
        "\n".join(
            [
                f"const engine = require({json.dumps(str(ENGINE))});",
                f"const golden = engine.parseGolden({json.dumps(raw_golden)});",
                f"const prediction = engine.parsePrediction({json.dumps(raw_prediction)});",
                "const heuristic = engine.heuristicPrediction(golden);",
                "const score = engine.scoreExample(golden, prediction);",
                "process.stdout.write(JSON.stringify({score: score, heuristic: heuristic}));",
            ]
        ),
        encoding="utf-8",
    )
    payload = json.loads(subprocess.check_output(["node", str(script)], text=True))
    js_score = payload["score"]
    for field in (
        "exact_match",
        "token_f1",
        "must_mention_recall",
        "forbidden_rate",
        "severity_accuracy",
        "tag_f1",
        "composite",
    ):
        assert js_score[field] == pytest.approx(getattr(python_score, field), rel=1e-9, abs=1e-12)
    assert payload["heuristic"]["prediction"] == prediction.prediction
    assert payload["heuristic"]["tags"] == list(prediction.tags)
