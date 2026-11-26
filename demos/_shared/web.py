"""A tiny JSON + static-file server (stdlib only) for the browser demos."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from plato import Event

Handler = Callable[[dict], object]


def event_json(ev: Event) -> dict:
    return {
        "seq": ev.seq, "id": ev.event_id, "artifact": ev.artifact_id, "actor_type": ev.actor_type,
        "actor": ev.actor_id, "surface": ev.surface, "op": ev.op, "parent_id": ev.parent_id,
        "text": ev.text, "payload": ev.payload, "at": ev.created_at,
    }


class App:
    def __init__(self, static_dir: Path) -> None:
        self.static_dir = Path(static_dir)
        self.routes: dict[tuple[str, str], Handler] = {}

    def get(self, path: str):
        return lambda fn: self.routes.setdefault(("GET", path), fn)

    def post(self, path: str):
        return lambda fn: self.routes.setdefault(("POST", path), fn)

    def serve(self, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the terminal for the agent
                pass

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                url = urlparse(self.path)
                fn = app.routes.get((method, url.path))
                if fn is not None:
                    params = {k: v[-1] for k, v in parse_qs(url.query).items()}
                    if method == "POST":
                        length = int(self.headers.get("Content-Length") or 0)
                        params.update(json.loads(self.rfile.read(length) or b"{}"))
                    try:
                        result = fn(params)
                    except (KeyError, ValueError) as e:
                        return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
                    return self._send(200, json.dumps(result).encode(), "application/json")
                if method == "GET":
                    name = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
                    f = (app.static_dir / name).resolve()
                    if f.is_file() and app.static_dir.resolve() in f.parents:
                        ctype = "text/html; charset=utf-8" if f.suffix == ".html" else "text/plain; charset=utf-8"
                        return self._send(200, f.read_bytes(), ctype)
                self._send(404, b"not found", "text/plain")

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        return server


def start_in_background(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t
