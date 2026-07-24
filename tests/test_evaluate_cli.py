import json
from pathlib import Path

from review_tuner.evaluate import main


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
