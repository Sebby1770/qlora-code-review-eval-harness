"""Local Review Tuner Studio — stdlib HTTP server, no extra deps."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from review_tuner import __version__
from review_tuner.data import load_examples, load_predictions
from review_tuner.evaluate import filter_examples, heuristic_prediction
from review_tuner.lint import lint_records
from review_tuner.metrics import render_html_report
from review_tuner.schema import DatasetError, Prediction, ReviewExample
from review_tuner.view import (
    GLOSSARY,
    METRIC_PLAIN,
    build_eval_view,
    compare_views,
    records_to_examples,
    records_to_predictions,
    summarize_dataset,
    unmatched_prediction_ids,
)

WEB_ROOT = Path(__file__).resolve().parent / "studio_web"
MAX_BODY_BYTES = 8 * 1024 * 1024
JSON_TYPE = "application/json; charset=utf-8"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        here.parents[2],
        here.parents[1],
    ]
    for candidate in candidates:
        if (candidate / "data" / "golden" / "code_review_golden.jsonl").is_file():
            return candidate
    return Path.cwd()


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("bundled sample dataset is missing")


def load_sample_golden() -> list[ReviewExample]:
    return load_examples(
        _first_existing(
            [
                WEB_ROOT / "samples" / "golden.jsonl",
                _repo_root() / "data" / "golden" / "code_review_golden.jsonl",
            ]
        )
    )


def load_sample_predictions() -> list[Prediction]:
    return load_predictions(
        _first_existing(
            [
                WEB_ROOT / "samples" / "predictions.sample.jsonl",
                _repo_root() / "examples" / "predictions.sample.jsonl",
            ]
        )
    )


def _json_bytes(payload: object, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return status, body, JSON_TYPE


def _parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise DatasetError("expected a JSON array or JSONL object stream")
            for index, item in enumerate(payload, start=1):
                if not isinstance(item, dict):
                    raise DatasetError(f"row {index}: expected a JSON object")
                records.append(item)
            return records
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"row {row_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"row {row_number}: expected a JSON object")
        records.append(value)
    return records


def _coerce_records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if value is None:
        raise DatasetError(f"{label} is required")
    if isinstance(value, list):
        records = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise DatasetError(f"{label} row {index}: expected a JSON object")
            records.append(item)
        return records
    if isinstance(value, str):
        return _parse_jsonl_text(value)
    raise DatasetError(f"{label} must be a JSON array or JSONL string")


def _token(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().lower() in {"sample", "bundled", "baseline"}:
        return value.strip().lower()
    return None


def _resolve_golden(value: Any) -> list[dict[str, Any]]:
    token = _token(value)
    if token in {"sample", "bundled"}:
        return [_example_to_record(example) for example in load_sample_golden()]
    if token == "baseline":
        raise DatasetError("golden cannot be 'baseline'; use sample or upload JSONL")
    return _coerce_records(value, label="golden")


def _resolve_predictions(value: Any, golden: list[ReviewExample]) -> list[Prediction]:
    token = _token(value)
    if value in (None, "", []) or token == "baseline":
        return [heuristic_prediction(example) for example in golden]
    if token in {"sample", "bundled"}:
        return load_sample_predictions()
    return records_to_predictions(_coerce_records(value, label="predictions"))


def _example_to_record(example: ReviewExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "language": example.language,
        "file_path": example.file_path,
        "context": example.context,
        "diff": example.diff,
        "expected_comment": example.target_comment,
        "severity": example.severity,
        "tags": list(example.tags),
        "rubric": {"must_mention": list(example.must_mention), "avoid": list(example.avoid)},
    }


def _prediction_to_record(prediction: Prediction) -> dict[str, Any]:
    return {
        "id": prediction.id,
        "prediction": prediction.prediction,
        "severity": prediction.severity,
        "tags": list(prediction.tags),
    }


_BADGE_COLORS = {
    "A": "#1f6f5b",
    "B": "#2f9a7c",
    "C": "#c4922a",
    "D": "#d46a52",
    "F": "#9b3b2a",
}


def render_badge(letter: str, verdict: str, percent: int) -> str:
    """Tiny SVG badge for READMEs and CI."""

    safe_letter = str(letter or "F")[:2]
    safe_verdict = str(verdict or "Fail")[:16]
    try:
        score = max(0, min(100, int(percent)))
    except (TypeError, ValueError):
        score = 0
    color = _BADGE_COLORS.get(safe_letter, "#5c6759")
    label = f"{safe_verdict} {safe_letter} {score}%"
    escaped = html.escape(label)
    width = 12 * len(label) + 24
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" '
        f'role="img" aria-label="{escaped}">'
        f'<rect width="{width}" height="20" rx="4" fill="#0e1a17"/>'
        f'<rect x="2" y="2" width="{width - 4}" height="16" rx="3" fill="{color}"/>'
        f'<text x="{width / 2:.1f}" y="14.5" text-anchor="middle" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="11" '
        f'fill="#fff">{escaped}</text></svg>\n'
    )


def _eval_view_from_body(body: dict[str, Any]) -> dict[str, Any]:
    golden_records = _resolve_golden(body.get("golden"))
    golden = records_to_examples(golden_records)
    golden = filter_examples(
        golden,
        language=body.get("language"),
        severity=body.get("severity"),
        tag=body.get("tag"),
    )
    predictions = _resolve_predictions(body.get("predictions"), golden)
    extra = unmatched_prediction_ids(golden, predictions)
    threshold = float(body.get("threshold") or 0.60)
    return build_eval_view(
        golden, predictions, threshold=threshold, extra_prediction_ids=extra
    )


def _safe_web_file(url_path: str) -> Path | None:
    relative = url_path.lstrip("/")
    if not relative or relative == "index.html":
        relative = "index.html"
    candidate = (WEB_ROOT / relative).resolve()
    try:
        candidate.relative_to(WEB_ROOT.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def handle_api(
    method: str, path: str, payload: Any, query: dict[str, list[str]] | None = None
) -> tuple[int, bytes, str]:
    """Dispatch JSON API routes. Used by the HTTP handler and tests."""

    if method == "GET" and path == "/api/health":
        return _json_bytes(
            {
                "ok": True,
                "name": "Review Tuner Studio",
                "version": __version__,
                "stdlib_only": True,
                "gpu_required": False,
            }
        )
    if method == "GET" and path == "/api/glossary":
        return _json_bytes({"glossary": GLOSSARY, "metrics": METRIC_PLAIN})
    if method == "GET" and path == "/api/samples/golden":
        records = [_example_to_record(example) for example in load_sample_golden()]
        return _json_bytes({"count": len(records), "records": records})
    if method == "GET" and path == "/api/samples/predictions":
        records = [_prediction_to_record(item) for item in load_sample_predictions()]
        return _json_bytes({"count": len(records), "records": records})
    if method == "GET" and path == "/api/badge":
        params = query or {}
        letter = (params.get("letter") or ["C"])[0]
        verdict_label = (params.get("verdict") or ["Weak"])[0]
        percent = (params.get("percent") or ["0"])[0]
        svg = render_badge(letter, verdict_label, percent)
        return 200, svg.encode("utf-8"), "image/svg+xml; charset=utf-8"
    if method == "GET" and path == "/api/samples/baseline":
        golden = load_sample_golden()
        records = [_prediction_to_record(heuristic_prediction(example)) for example in golden]
        return _json_bytes({"count": len(records), "records": records, "source": "baseline"})

    if method != "POST":
        return _json_bytes({"error": "not found"}, 404)

    body = payload if isinstance(payload, dict) else {}

    if path == "/api/eval":
        view = _eval_view_from_body(body)
        return _json_bytes(view)

    if path == "/api/baseline":
        golden = records_to_examples(_resolve_golden(body.get("golden")))
        records = [_prediction_to_record(heuristic_prediction(example)) for example in golden]
        return _json_bytes({"count": len(records), "records": records, "source": "baseline"})

    if path == "/api/lint":
        if body.get("text"):
            try:
                records = [
                    (index, item)
                    for index, item in enumerate(_parse_jsonl_text(str(body["text"])), start=1)
                ]
            except DatasetError as exc:
                return _json_bytes(
                    {
                        "ok": False,
                        "errors": 1,
                        "issues": [
                            {
                                "level": "error",
                                "rule": "invalid_json",
                                "message": str(exc),
                            }
                        ],
                    },
                    200,
                )
        else:
            spec = body.get("records")
            if spec is None:
                spec = body.get("golden")
            if _token(spec) in {"sample", "bundled"}:
                raw_records = _resolve_golden("sample")
            else:
                raw_records = _coerce_records(spec, label="records")
            records = list(enumerate(raw_records, start=1))
        issues = lint_records(records)
        errors = [issue for issue in issues if issue.level == "error"]
        coverage = summarize_dataset([item for _row, item in records])
        return _json_bytes(
            {
                "ok": not errors,
                "errors": len(errors),
                "warnings": sum(1 for issue in issues if issue.level == "warning"),
                "coverage": coverage,
                "issues": [
                    {
                        "level": issue.level,
                        "rule": issue.rule,
                        "message": issue.message,
                        "id": issue.example_id,
                        "row": issue.row,
                    }
                    for issue in issues
                ],
            }
        )

    if path == "/api/compare":
        golden_records = _resolve_golden(body.get("golden"))
        golden = records_to_examples(golden_records)
        pred_a = _resolve_predictions(body.get("predictions_a"), golden)
        pred_b = _resolve_predictions(body.get("predictions_b"), golden)
        threshold = float(body.get("threshold") or 0.60)
        left = build_eval_view(golden, pred_a, threshold=threshold)
        right = build_eval_view(golden, pred_b, threshold=threshold)
        report = compare_views(left, right)
        report["a_view"] = left
        report["b_view"] = right
        return _json_bytes(report)

    if path == "/api/report-html":
        view = _eval_view_from_body(body)
        html = render_html_report(
            view["aggregate"], view["examples"], view["error_analysis"]
        )
        return 200, html.encode("utf-8"), "text/html; charset=utf-8"

    if path == "/api/badge":
        view = _eval_view_from_body(body)
        badge = view.get("badge") or {}
        svg = render_badge(
            str(badge.get("letter") or view.get("letter_grade") or "F"),
            str(badge.get("verdict") or view.get("verdict") or "Fail"),
            int(badge.get("percent") or 0),
        )
        return 200, svg.encode("utf-8"), "image/svg+xml; charset=utf-8"

    return _json_bytes({"error": "not found"}, 404)


class StudioHandler(BaseHTTPRequestHandler):
    server_version = f"ReviewTunerStudio/{__version__}"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise DatasetError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON body: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            try:
                status, body, content_type = handle_api(
                    "GET", path, None, parse_qs(parsed.query)
                )
            except (DatasetError, FileNotFoundError, OSError, ValueError) as exc:
                status, body, content_type = _json_bytes({"error": str(exc)}, 400)
            self._send(status, body, content_type)
            return
        target = _safe_web_file(path)
        if target is None:
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or target.suffix in {".js", ".css", ".svg"}:
            content_type = f"{content_type}; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            status, body, content_type = handle_api("POST", parsed.path, payload)
        except (DatasetError, json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
            status, body, content_type = _json_bytes({"error": str(exc)}, 400)
        self._send(status, body, content_type)


def serve(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> None:
    if not (WEB_ROOT / "index.html").is_file():
        raise FileNotFoundError(f"studio frontend missing at {WEB_ROOT}")
    httpd = ThreadingHTTPServer((host, port), StudioHandler)
    url = f"http://{host}:{port}/"
    print(f"Review Tuner Studio {__version__}")
    print(f"Open {url}")
    print("This server is local-first. It does not send your diffs anywhere.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping studio")
    finally:
        httpd.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-eval studio",
        description="Open the local Review Tuner Studio in your browser.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the studio in a browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.port < 1 or args.port > 65535:
        print("error: port must be between 1 and 65535", file=sys.stderr)
        return 2
    try:
        serve(args.host, args.port, open_browser=args.open_browser)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
