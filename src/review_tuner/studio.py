"""Serve the local static evaluation studio without extra dependencies."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def studio_root() -> Path:
    """Return the packaged studio_web directory."""

    return Path(__file__).resolve().parent / "studio_web"


class StudioHandler(SimpleHTTPRequestHandler):
    """Serve files from studio_web, including nested sample JSONL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(studio_root()), **kwargs)


def bind_server(host: str, port: int) -> ThreadingHTTPServer:
    """Bind a threaded static file server to host:port (port 0 chooses free)."""

    root = studio_root()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(f"studio assets missing at {root}")
    return ThreadingHTTPServer((host, port), StudioHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="review-eval studio",
        description="Serve the local web studio (static files, no extra deps).",
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port. Default: 8765.")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the studio in the default browser.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1.")
    args = parser.parse_args(argv)
    try:
        server = bind_server(args.host, args.port)
    except OSError as exc:
        print(f"could not bind studio server: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    raw_host, raw_port = server.server_address[0], server.server_address[1]
    host = raw_host.decode("ascii") if isinstance(raw_host, bytes) else raw_host
    url = f"http://{host}:{int(raw_port)}/"
    print(f"serving review studio at {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping studio")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
