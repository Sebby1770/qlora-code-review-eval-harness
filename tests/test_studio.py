from __future__ import annotations

import threading
from urllib.request import urlopen

from review_tuner.studio import bind_server, main, studio_root


def test_studio_root_contains_static_assets() -> None:
    root = studio_root()
    assert (root / "index.html").is_file()
    assert (root / "eval-engine.js").is_file()
    assert (root / "app.js").is_file()
    assert (root / "styles.css").is_file()
    assert (root / "samples" / "golden.jsonl").is_file()
    assert (root / "samples" / "predictions.jsonl").is_file()


def test_studio_server_serves_index_and_samples() -> None:
    server = bind_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:
            homepage = response.read().decode("utf-8")
        with urlopen(f"http://{host}:{port}/eval-engine.js", timeout=2) as response:
            engine = response.read().decode("utf-8")
        with urlopen(f"http://{host}:{port}/samples/golden.jsonl", timeout=2) as response:
            sample = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "Night Desk" in homepage
    assert "0.35" in engine
    assert "golden-001" in sample


def test_studio_main_returns_after_shutdown(monkeypatch) -> None:
    calls: list[str] = []

    def fake_serve_forever(_self: object) -> None:
        calls.append("serve")
        raise KeyboardInterrupt

    monkeypatch.setattr("review_tuner.studio.ThreadingHTTPServer.serve_forever", fake_serve_forever)
    assert main(["--port", "0"]) == 0
    assert calls == ["serve"]
