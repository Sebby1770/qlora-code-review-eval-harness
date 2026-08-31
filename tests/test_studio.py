import json
from pathlib import Path

from review_tuner.evaluate import main
from review_tuner.studio import handle_api


def _post(path: str, payload: dict):
    status, body, content_type = handle_api("POST", path, payload)
    data = json.loads(body.decode("utf-8"))
    return status, data, content_type


def _get(path: str):
    status, body, content_type = handle_api("GET", path, None)
    return status, json.loads(body.decode("utf-8")), content_type


def test_health_and_samples():
    status, data, _ = _get("/api/health")
    assert status == 200
    assert data["ok"] is True
    assert data["stdlib_only"] is True
    status, golden, _ = _get("/api/samples/golden")
    assert status == 200
    assert golden["count"] == 15
    status, preds, _ = _get("/api/samples/predictions")
    assert preds["count"] == 15


def test_badge_svg_from_eval_and_query():
    status, body, content_type = handle_api(
        "POST",
        "/api/badge",
        {"golden": "sample", "predictions": "sample"},
    )
    assert status == 200
    assert "svg" in content_type
    svg = body.decode("utf-8")
    assert "<svg" in svg
    assert "Weak" in svg or "Pass" in svg or "Fail" in svg
    status, body, content_type = handle_api(
        "GET",
        "/api/badge",
        None,
        {"letter": ["A"], "verdict": ["Pass"], "percent": ["91"]},
    )
    assert status == 200
    assert "Pass A 91%" in body.decode("utf-8")


def test_report_html_api():
    status, body, content_type = handle_api(
        "POST",
        "/api/report-html",
        {"golden": "sample", "predictions": "sample"},
    )
    assert status == 200
    assert "text/html" in content_type
    html = body.decode("utf-8")
    assert "<details" in html
    assert "Example inspector" in html


def test_eval_api_includes_histogram():
    status, view, _ = _post("/api/eval", {"golden": "sample", "predictions": "sample"})
    assert status == 200
    assert sum(view["histogram"].values()) == 15
    assert "security" in view["by_tag"]


def test_eval_api_accepts_sample_tokens():
    status, view, _ = _post("/api/eval", {"golden": "sample", "predictions": "sample"})
    assert status == 200
    assert view["aggregate"]["count"] == 15
    assert abs(float(view["aggregate"]["composite"]) - 0.5438) < 0.02


def test_eval_api_baseline_token():
    status, view, _ = _post("/api/eval", {"golden": "sample", "predictions": "baseline"})
    assert status == 200
    assert view["aggregate"]["count"] == 15


def test_lint_sample_includes_coverage():
    status, result, _ = _post("/api/lint", {"records": "sample"})
    assert status == 200
    assert result["ok"] is True
    assert result["coverage"]["count"] == 15
    assert "python" in result["coverage"]["languages"]
    assert "security" in result["coverage"]["tags"]


def test_compare_sample_vs_baseline_tokens():
    status, report, _ = _post(
        "/api/compare",
        {
            "golden": "sample",
            "predictions_a": "baseline",
            "predictions_b": "sample",
        },
    )
    assert status == 200
    assert "delta_b_minus_a" in report
    assert "a_view" in report and "b_view" in report


def test_eval_api_sample_includes_diff_and_story():
    _, golden, _ = _get("/api/samples/golden")
    _, preds, _ = _get("/api/samples/predictions")
    status, view, _ = _post(
        "/api/eval",
        {"golden": golden["records"], "predictions": preds["records"]},
    )
    assert status == 200
    assert "story" in view
    assert view["examples"][0]["diff"]
    assert "explanation" in view["examples"][0]
    assert abs(float(view["aggregate"]["composite"]) - 0.5438) < 0.02


def test_eval_api_baseline_without_predictions():
    _, golden, _ = _get("/api/samples/golden")
    status, view, _ = _post("/api/eval", {"golden": golden["records"]})
    assert status == 200
    assert view["aggregate"]["count"] == 15


def test_lint_api_duplicate_ids():
    status, result, _ = _post(
        "/api/lint",
        {
            "records": [
                {
                    "id": "dup",
                    "diff": "x",
                    "file_path": "a.py",
                    "language": "python",
                    "context": "c",
                    "expected_comment": "comment",
                    "severity": "high",
                    "tags": ["tests"],
                    "rubric": {"must_mention": ["x"], "avoid": []},
                },
                {
                    "id": "dup",
                    "diff": "y",
                    "file_path": "a.py",
                    "language": "python",
                    "context": "c",
                    "expected_comment": "comment",
                    "severity": "high",
                    "tags": ["tests"],
                    "rubric": {"must_mention": ["y"], "avoid": []},
                },
            ]
        },
    )
    assert status == 200
    assert result["ok"] is False
    assert result["errors"] >= 1
    assert any(issue["rule"] == "duplicate_id" for issue in result["issues"])


def test_compare_api_delta_zero_for_identical():
    _, golden, _ = _get("/api/samples/golden")
    _, baseline, _ = _get("/api/samples/baseline")
    status, report, _ = _post(
        "/api/compare",
        {
            "golden": golden["records"],
            "predictions_a": baseline["records"],
            "predictions_b": baseline["records"],
        },
    )
    assert status == 200
    assert abs(report["delta_b_minus_a"]["composite"]) < 1e-9


def test_html_report_includes_inspector(tmp_path: Path):
    html_path = tmp_path / "report.html"
    code = main(
        [
            "eval",
            "--golden",
            "data/golden/code_review_golden.jsonl",
            "--predictions",
            "examples/predictions.sample.jsonl",
            "--out",
            str(tmp_path / "eval.json"),
            "--per-example-out",
            str(tmp_path / "per.jsonl"),
            "--report-html",
            str(html_path),
        ]
    )
    assert code == 0
    html = html_path.read_text(encoding="utf-8")
    assert "<details" in html
    assert "Example inspector" in html
    assert "expired" in html.lower()


def test_studio_help_and_frontend_packaged():
    from review_tuner.studio import WEB_ROOT, build_parser

    help_text = build_parser().format_help()
    assert "--port" in help_text
    assert (WEB_ROOT / "index.html").is_file()
    assert (WEB_ROOT / "styles.css").is_file()
    assert (WEB_ROOT / "app.js").is_file()
    assert (WEB_ROOT / "samples" / "golden.jsonl").is_file()
