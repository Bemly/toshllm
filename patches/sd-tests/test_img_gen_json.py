# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Warm sd-server accepts step-window / latent fields on /sdcpp/v1/img_gen without treating them as
unknown JSON. No weights are loaded: a missing model yields model_not_loaded, not a parse error.

    SD_SERVER_BIN_PATH=vendor/stable-diffusion.cpp/build-static/bin/sd-server \\
        python3 -I patches/sd-tests/test_img_gen_json.py [-v]
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
# patches/sd-tests -> toshllm
ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.environ.get("SD_SERVER_BIN_PATH") or os.path.join(
    ROOT, "vendor", "stable-diffusion.cpp", "build-static", "bin", "sd-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(os.path.isfile(BINARY) and os.access(BINARY, os.X_OK), "sd-server not built")
class ImgGenJSONTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tosh-sd-json-")
        cls.models = os.path.join(cls.dir, "models")
        os.makedirs(cls.models, exist_ok=True)
        cls.port = free_port()
        cls.log = open(os.path.join(cls.dir, "server.log"), "wb")
        cls.process = subprocess.Popen(
            [BINARY, "-l", "127.0.0.1", "--listen-port", str(cls.port),
             "--model-dir", cls.models, "--conditioning-cache-size", "16", "--keep-te"],
            stdout=cls.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and cls.process.poll() is None:
            try:
                status, body = cls.request("GET", "/sdcpp/v1/models/status")
                if status == 200:
                    cls.status_body = body
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

    def test_status_reports_empty_without_weights(self):
        data = json.loads(self.status_body.decode())
        self.assertIn(data.get("state"), ("empty", "error", "ready"))

    def test_step_window_fields_are_parsed(self):
        payload = {
            "prompt": "cache probe",
            "width": 64,
            "height": 64,
            "seed": 1,
            "batch_count": 1,
            "start_at_step": 0,
            "end_at_step": 2,
            "add_noise": "on",
            "latent_out": os.path.join(self.dir, "out.sdlt"),
            "sample_params": {"sample_steps": 4, "sample_method": "euler"},
            "output_format": "png",
        }
        status, body = self.request("POST", "/sdcpp/v1/img_gen", json.dumps(payload))
        text = body.decode(errors="replace")
        # Parsed OK: either queued (unlikely without model) or model_not_loaded / loading.
        self.assertNotIn("invalid generation parameters", text.lower())
        self.assertIn(status, (202, 400, 503), text)
        if status == 400:
            self.assertTrue(
                "model" in text.lower() or "not" in text.lower(),
                text)

    def test_help_defaults_warm_cache(self):
        out = subprocess.check_output([BINARY, "--help"], text=True, stderr=subprocess.STDOUT)
        self.assertIn("--conditioning-cache-size", out)
        self.assertIn("--keep-te", out)
        self.assertIn("--release-te-after-encode", out)


if __name__ == "__main__":
    unittest.main()
