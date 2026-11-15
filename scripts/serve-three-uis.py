#!/usr/bin/env python3
"""Serve three static WebUIs on three ports; proxy API to one llama-server.

Ports (defaults):
  8080  llama.cpp tools/ui (or --llama-ui)
  8081  kvmem ui (lightweight + monitor) (--kvmem-ui)
  8082  HyperCore serve/web (--hypercore-ui)

All non-static requests are proxied to --backend (default http://127.0.0.1:9931).

Example:
  ./scripts/serve-three-uis.py \\
    --backend http://127.0.0.1:9931 \\
    --llama-ui /path/to/llama-ui-dist \\
    --kvmem-ui /path/to/kvmem/ui-dist \\
    --hypercore-ui /path/to/HyperCore-WebUI/serve/web
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import os
import socketserver
import threading
import urllib.parse
from pathlib import Path


def parse_backend(url: str) -> tuple[str, int, str]:
    u = urllib.parse.urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, port, u.scheme or "http"


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    backend_host = "127.0.0.1"
    backend_port = 9931
    static_root: Path = Path(".")

    def translate_path(self, path: str) -> str:
        # serve only under static_root
        parsed = urllib.parse.urlparse(path)
        rel = parsed.path.lstrip("/")
        if not rel or rel.endswith("/"):
            rel = (rel + "index.html") if rel else "index.html"
        candidate = (self.static_root / rel).resolve()
        root = self.static_root.resolve()
        if not str(candidate).startswith(str(root)):
            return str(root / "index.html")
        return str(candidate)

    def _is_static(self) -> bool:
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", ""):
            return True
        # API-ish paths go to backend
        api_prefixes = (
            "/v1/",
            "/api/",
            "/health",
            "/props",
            "/metrics",
            "/slots",
            "/models",
            "/prompts",
            "/history/",
            "/settings",
            "/config",
            "/mcp",
            "/assist/",
            "/tune/",
            "/build/",
            "/versions",
            "/load",
            "/unload",
            "/completion",
            "/completions",
            "/chat/",
            "/responses",
            "/tokenize",
            "/detokenize",
            "/embedding",
            "/embeddings",
            "/lora-adapters",
            "/infill",
            "/tools",
            "/cors-proxy",
        )
        for p in api_prefixes:
            if path == p.rstrip("/") or path.startswith(p):
                return False
        # hashed assets / pages
        return True

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else None
        conn = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except Exception as e:
            msg = f'{{"error":{{"message":"proxy failed: {e}"}}}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
        finally:
            conn.close()

    def do_GET(self) -> None:  # noqa: N802
        if self._is_static():
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        return self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._is_static():
            return http.server.SimpleHTTPRequestHandler.do_HEAD(self)
        return self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        return self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        return self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        return self._proxy()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.server.server_address[1]}] {self.address_string()} {fmt % args}")


def serve(port: int, root: Path, backend: str) -> None:
    host, bport, _ = parse_backend(backend)

    class Handler(ProxyHandler):
        pass

    Handler.backend_host = host
    Handler.backend_port = bport
    Handler.static_root = root
    Handler.directory = str(root)

    with socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"UI http://127.0.0.1:{port}/  root={root}  -> {backend}")
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="http://127.0.0.1:9931")
    ap.add_argument("--llama-ui", type=Path, required=True)
    ap.add_argument("--kvmem-ui", type=Path, required=True)
    ap.add_argument("--hypercore-ui", type=Path, required=True)
    ap.add_argument("--llama-port", type=int, default=8080)
    ap.add_argument("--kvmem-port", type=int, default=8081)
    ap.add_argument("--hypercore-port", type=int, default=8082)
    args = ap.parse_args()

    for p in (args.llama_ui, args.kvmem_ui, args.hypercore_ui):
        if not p.is_dir():
            raise SystemExit(f"missing UI dir: {p}")

    threads = [
        threading.Thread(target=serve, args=(args.llama_port, args.llama_ui.resolve(), args.backend), daemon=True),
        threading.Thread(target=serve, args=(args.kvmem_port, args.kvmem_ui.resolve(), args.backend), daemon=True),
        threading.Thread(
            target=serve, args=(args.hypercore_port, args.hypercore_ui.resolve(), args.backend), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    print("three UI ports up; Ctrl+C to stop")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
