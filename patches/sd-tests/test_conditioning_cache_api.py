# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Warm sd-server exposes conditioning-cache list / clear / purge without a loaded model.

    SD_SERVER_BIN_PATH=vendor/stable-diffusion.cpp/build-static/bin/sd-server \\
        python3 -I patches/sd-tests/test_conditioning_cache_api.py [-v]
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
BINARY = os.environ.get("SD_SERVER_BIN_PATH") or os.path.join(
    ROOT, "vendor", "stable-diffusion.cpp", "build-static", "bin", "sd-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(os.path.isfile(BINARY) and os.access(BINARY, os.X_OK), "sd-server not built")
class ConditioningCacheAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tosh-sd-cache-")
        cls.models = os.path.join(cls.dir, "models")
        os.makedirs(cls.models, exist_ok=True)
        cls.port = free_port()
        cls.log = open(os.path.join(cls.dir, "server.log"), "wb")
        cls.process = subprocess.Popen(
            [BINARY, "-l", "127.0.0.1", "--listen-port", str(cls.port),
             "--model-dir", cls.models, "--conditioning-cache-size", "16"],
            stdout=cls.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and cls.process.poll() is None:
            try:
                status, _ = cls.request("GET", "/sdcpp/v1/models/status")
                if status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("sd-server did not start; see " + cls.log.name)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(15)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(5)
        cls.log.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def request(cls, method, path, body=None, timeout=10):
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=timeout)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            res = conn.getresponse()
            return res.status, res.read()
        finally:
            conn.close()

    def test_list_without_model(self):
        status, body = self.request("GET", "/sdcpp/v1/conditioning-cache")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("entries"), [])
        self.assertFalse(data.get("model_loaded"))

    def test_purge_without_model_is_503(self):
        status, body = self.request(
            "POST", "/sdcpp/v1/conditioning-cache/purge",
            body=json.dumps({"max_age_seconds": 86400}))
        self.assertEqual(status, 503)
        data = json.loads(body)
        self.assertEqual(data.get("code"), "model_not_loaded")

    def test_delete_without_model_is_503(self):
        status, body = self.request("DELETE", "/sdcpp/v1/conditioning-cache")
        self.assertEqual(status, 503)
        data = json.loads(body)
        self.assertEqual(data.get("code"), "model_not_loaded")


if __name__ == "__main__":
    unittest.main()
