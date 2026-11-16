# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compat monitor schema: /metrics Accept split, KVMem snapshot field names, request ledger
on /v1/chat/completions. Router mode (no model) still exposes the APIs.

    LLAMA_SERVER_BIN_PATH=vendor/llama.cpp/build-static/bin/llama-server \\
        python3 -I patches/llama-tests/test_compat_metrics.py [-v]
"""

import http.client
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.environ.get("LLAMA_SERVER_BIN_PATH") or os.path.join(
    ROOT, "vendor", "llama.cpp", "build-static", "bin", "llama-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CompatMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tosh-compat-metrics-")
        cls.port = free_port()
        cls.log = open(os.path.join(cls.dir, "server.log"), "wb")
        cls.process = subprocess.Popen(
            [BINARY, "--host", "127.0.0.1", "--port", str(cls.port), "--offline", "--metrics"],
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
    def request(cls, method, path, body=None, headers=None, timeout=5):
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=timeout)
        try:
            payload = None if body is None else (
                body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode())
            hdrs = dict(headers or {})
            if payload is not None and "Content-Type" not in hdrs:
                hdrs["Content-Type"] = "application/json"
            conn.request(method, path, body=payload, headers=hdrs)
            res = conn.getresponse()
            data = res.read()
            out_headers = {k.lower(): v for k, v in res.getheaders()}
            return res.status, data, out_headers
        finally:
            conn.close()

    def test_metrics_default_is_hypercore_json(self):
        status, body, _ = self.request("GET", "/metrics")
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        self.assertIn("hardware", data)
        self.assertIn("live", data)
        self.assertIn("cpu", data["hardware"])
        self.assertIn("gpu_mem_used", data["hardware"])
        self.assertIn("gpus", data["hardware"])

    def test_metrics_accept_text_plain_is_prometheus(self):
        status, body, headers = self.request(
            "GET", "/metrics", headers={"Accept": "text/plain"})
        self.assertEqual(status, 200, body)
        text = body.decode("utf-8", "replace")
        self.assertTrue(
            text.startswith("# HELP") or "llamacpp:" in text,
            text[:200])
        ctype = headers.get("content-type", "")
        self.assertIn("text/plain", ctype)

    def test_metrics_prom_alias(self):
        status, body, _ = self.request("GET", "/metrics.prom")
        self.assertEqual(status, 200, body)
        self.assertIn(b"llamacpp:", body)

    def test_snapshot_uses_kvmem_gpu_field_names(self):
        status, body, _ = self.request("GET", "/api/monitor/snapshot")
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        self.assertEqual(data.get("schema_version"), 1)
        hw = data["hardware"]
        self.assertIn("cpu_percent", hw)
        self.assertIn("ram_total_bytes", hw)
        gpus = hw.get("gpus") or []
        self.assertTrue(gpus, data)
        gpu = gpus[0]
        for key in ("utilization_percent", "used_bytes", "total_bytes", "temperature_c"):
            self.assertIn(key, gpu, gpu)

    def test_launcher_hardware_has_flat_ram_aliases(self):
        status, body, _ = self.request("GET", "/api/launcher/hardware")
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        self.assertEqual(data.get("scope"), "system")
        self.assertIn("ram", data)
        self.assertIn("ram_total_bytes", data)

    def test_v1_chat_is_recorded_even_without_model(self):
        before = json.loads(self.request("GET", "/api/monitor/snapshot")[1])
        n_before = len(before.get("recent") or [])
        status, body, _ = self.request(
            "POST", "/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        # no model loaded -> error, but ledger should still finish the row
        self.assertGreaterEqual(status, 400, body)
        deadline = time.monotonic() + 5
        recent = []
        while time.monotonic() < deadline:
            snap = json.loads(self.request("GET", "/api/monitor/snapshot")[1])
            recent = snap.get("recent") or []
            if len(recent) > n_before:
                break
            time.sleep(0.1)
        self.assertGreater(len(recent), n_before, recent)
        metrics = json.loads(self.request("GET", "/metrics")[1])
        self.assertGreaterEqual(len(metrics.get("requests") or []), 1, metrics)


if __name__ == "__main__":
    unittest.main()
