# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-port --ui-mount for /kvmem and /strata: bare paths redirect with a trailing slash so
relative CSS/JS stay under the mount, and those assets return 200. No model: the engine runs as a
router.

    LLAMA_SERVER_BIN_PATH=vendor/llama.cpp/build-static/bin/llama-server \\
        python3 -I patches/llama-tests/test_ui_mounts.py [-v]
"""

import http.client
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# patches/llama-tests -> toshllm
ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.environ.get("LLAMA_SERVER_BIN_PATH") or os.path.join(
    ROOT, "vendor", "llama.cpp", "build-static", "bin", "llama-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_tree(root, name, rel_asset):
    base = os.path.join(root, name)
    asset_dir = os.path.join(base, os.path.dirname(rel_asset)) if os.path.dirname(rel_asset) else base
    os.makedirs(asset_dir, exist_ok=True)
    with open(os.path.join(base, "index.html"), "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><link rel="stylesheet" href="{rel_asset}">\n')
    with open(os.path.join(base, rel_asset), "w", encoding="utf-8") as f:
        f.write(f"/* {name} */\n")
    return base


class UIMountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tosh-ui-mount-")
        cls.web = write_tree(cls.dir, "web-ui", "main.css")
        cls.kvmem = write_tree(cls.dir, "kvmem", "launcher.css")
        cls.strata = write_tree(cls.dir, "strata", "web/tokens.css")
        cls.port = free_port()
        mounts = f"/kvmem:{cls.kvmem},/monitor:{cls.kvmem},/strata:{cls.strata}"
        cls.log = open(os.path.join(cls.dir, "server.log"), "wb")
        cls.process = subprocess.Popen(
            [BINARY, "--host", "127.0.0.1", "--port", str(cls.port), "--offline",
             "--path", cls.web, "--ui-mount", mounts],
            stdout=cls.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and cls.process.poll() is None:
            try:
                if cls.request("GET", "/health")[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become ready; see " + cls.log.name)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(20)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(5)
        cls.log.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def request(cls, method, path, timeout=5):
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=timeout)
        try:
            conn.request(method, path)
            res = conn.getresponse()
            body = res.read()
            headers = {k.lower(): v for k, v in res.getheaders()}
            return res.status, body, headers
        finally:
            conn.close()

    def test_bare_prefix_redirects_to_trailing_slash(self):
        for prefix in ("/kvmem", "/monitor", "/strata"):
            with self.subTest(prefix=prefix):
                status, _, headers = self.request("GET", prefix)
                self.assertIn(status, (301, 302, 303, 307, 308), status)
                location = headers.get("location", "")
                self.assertTrue(
                    location == prefix + "/" or location.endswith(prefix + "/"),
                    location)

    def test_assets_under_mount_are_served(self):
        cases = (
            ("/kvmem/launcher.css", b"kvmem"),
            ("/monitor/launcher.css", b"kvmem"),
            ("/strata/web/tokens.css", b"strata"),
            ("/main.css", b"web-ui"),
        )
        for path, marker in cases:
            with self.subTest(path=path):
                status, body, _ = self.request("GET", path)
                self.assertEqual(status, 200, body)
                self.assertIn(marker, body)

    def test_compat_monitor_snapshot_exists(self):
        status, body, _ = self.request("GET", "/api/monitor/snapshot")
        self.assertEqual(status, 200, body)
        self.assertIn(b"schema", body.lower() + body)


if __name__ == "__main__":
    unittest.main()
