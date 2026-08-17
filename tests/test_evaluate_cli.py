import json
from pathlib import Path

from review_tuner.evaluate import deleted_line_tokens, heuristic_prediction, main
from review_tuner.schema import ReviewExample


def test_eval_baseline(tmp_path: Path):
    golden = Path("data/golden/code_review_golden.jsonl")
    out = tmp_path / "eval.json"
    per = tmp_path / "per.jsonl"
    md = tmp_path / "report.md"
    code = main(
        [
            "eval",
            "--golden",
            str(golden),
            "--out",
            str(out),
            "--per-example-out",
            str(per),
            "--report-md",
            str(md),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert data["count"] >= 4
    assert "composite" in data
    assert md.exists()
    assert "Metrics" in md.read_text()


def test_baseline_and_validate(tmp_path: Path):
    golden = "data/golden/code_review_golden.jsonl"
    pred = tmp_path / "base.jsonl"
    assert main(["baseline", "--golden", golden, "--out", str(pred)]) == 0
    assert pred.exists()
    assert main(["validate", golden]) == 0


def test_compare(tmp_path: Path):
    golden = "data/golden/code_review_golden.jsonl"
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    assert main(["baseline", "--golden", golden, "--out", str(a)]) == 0
    # second file identical → delta ~ 0
    b.write_text(a.read_text())
    out = tmp_path / "cmp.json"
    assert main(["compare", "--golden", golden, str(a), str(b), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert "delta_b_minus_a" in report
    assert abs(report["delta_b_minus_a"]["composite"]) < 1e-9


def test_eval_html_and_json_reports(tmp_path: Path):
    golden = Path("data/golden/code_review_golden.jsonl")
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    code = main(
        [
            "eval",
            "--golden",
            str(golden),
            "--out",
            str(tmp_path / "eval.json"),
            "--per-example-out",
            str(tmp_path / "per.jsonl"),
            "--report-html",
            str(html_path),
            "--report-json",
            str(json_path),
        ]
    )
    assert code == 0
    html = html_path.read_text(encoding="utf-8")
    assert "composite" in html.lower()
    assert "golden-001" in html
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "error_analysis" in payload
    assert "composite_ci" in payload


def test_dashboard_writes_index(tmp_path: Path):
    (tmp_path / "eval.json").write_text(
        json.dumps({"count": 2, "composite": 0.42, "composite_ci_lo": 0.3, "composite_ci_hi": 0.5}),
        encoding="utf-8",
    )
    (tmp_path / "eval.md").write_text("# Code review eval report\n", encoding="utf-8")
    assert main(["dashboard", str(tmp_path)]) == 0
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "composite" in html.lower()
    assert "eval.json" in html


def test_heuristic_mentions_deleted_line_tokens():
    example = ReviewExample(
        id="deleted",
        diff="-    rows = db.execute('SELECT id FROM users WHERE email = ?', (email,))\n"
        "+    rows = db.execute(f\"SELECT id FROM users WHERE email = '{email}'\")",
        file_path="api/search.py",
        language="python",
        context="parameterized lookup",
        target_comment="keep parameterized SQL",
        severity="high",
        tags=("security",),
    )
    tokens = deleted_line_tokens(example.diff)
    assert "email" in [token.lower() for token in tokens]
    assert "---" not in tokens
    first = heuristic_prediction(example)
    second = heuristic_prediction(example)
    assert first.prediction == second.prediction
    lowered = first.prediction.lower()
    assert "email" in lowered
    assert "users" in lowered or "select" in lowered
