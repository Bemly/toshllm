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
        cls.cache = os.path.join(cls.dir, "cache")
        os.makedirs(cls.cache, exist_ok=True)
        os.makedirs(cls.models, exist_ok=True)
        cls.port = free_port()
        cls.log = open(os.path.join(cls.dir, "server.log"), "wb")
        cls.process = subprocess.Popen(
            [BINARY, "-l", "127.0.0.1", "--listen-port", str(cls.port),
             "--model-dir", cls.models, "--conditioning-cache-size", "16",
             "--conditioning-cache-dir", cls.cache],
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

    def test_disk_delete_without_model(self):
        scope, entry = "0123456789abcdef", "fedcba9876543210"
        folder = os.path.join(self.cache, scope)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, entry + ".cndcache")
        role_path = os.path.join(folder, entry + ".roles")
        with open(path, "wb") as file:
            file.write(b"cache")
        with open(role_path, "w") as file:
            file.write("1")
        status, _ = self.request("DELETE", "/sdcpp/v1/conditioning-cache/disk?scope=..&id=" + entry)
        self.assertEqual(status, 400)
        self.assertTrue(os.path.exists(path))
        status, _ = self.request("DELETE", f"/sdcpp/v1/conditioning-cache/disk?scope={scope}&id={entry}")
        self.assertEqual(status, 200)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(role_path))
        with open(path, "wb") as file:
            file.write(b"cache")
        with open(role_path, "w") as file:
            file.write("2")
        status, _ = self.request("DELETE", "/sdcpp/v1/conditioning-cache/disk?all=true")
        self.assertEqual(status, 200)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(role_path))


@unittest.skipUnless(
    os.path.isfile(BINARY) and os.access(BINARY, os.X_OK) and
    all(os.environ.get(key) for key in
        ("SD_CACHE_TEST_MODEL", "SD_CACHE_TEST_VAE", "SD_CACHE_TEST_LLM", "SD_CACHE_TEST_TOKENIZER")),
    "real model paths not provided")
class ConditioningCacheLiveTest(unittest.TestCase):
    def test_cache_snapshot_during_generation(self):
        port = free_port()
        with tempfile.TemporaryDirectory(prefix="tosh-sd-cache-live-") as directory:
            process = subprocess.Popen(
                [BINARY, "-l", "127.0.0.1", "--listen-port", str(port),
                 "--conditioning-cache-size", "16",
                 "--conditioning-cache-dir", directory,
                 "--diffusion-model", os.environ["SD_CACHE_TEST_MODEL"],
                 "--vae", os.environ["SD_CACHE_TEST_VAE"],
                 "--llm", os.environ["SD_CACHE_TEST_LLM"],
                 "--tokenizer", os.environ["SD_CACHE_TEST_TOKENIZER"],
                 "--release-te-after-encode", "--auto-fit", "off"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            def request(method, path, body=None, timeout=3):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
                try:
                    headers = {"Content-Type": "application/json"} if body is not None else {}
                    conn.request(method, path, body=body, headers=headers)
                    response = conn.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    conn.close()

            try:
                deadline = time.monotonic() + 180
                while True:
                    self.assertIsNone(process.poll(), "server exited during startup")
                    try:
                        status, _ = request("GET", "/sdcpp/v1/models/status")
                        if status == 200:
                            break
                    except OSError:
                        pass
                    self.assertLess(time.monotonic(), deadline, "server startup timed out")
                    time.sleep(0.5)

                body = json.dumps({
                    "prompt": "a blue ceramic cup on a table", "negative_prompt": "blurry",
                    "width": 256, "height": 256, "seed": 123, "batch_count": 1,
                    "sample_params": {
                        "scheduler": "sgm_uniform", "sample_method": "euler",
                        "sample_steps": 4, "flow_shift": 5.0, "guidance": {"txt_cfg": 3.0},
                    },
                    "preview": "none",
                })
                status, job = request("POST", "/sdcpp/v1/img_gen", body)
                self.assertEqual(status, 202)
                poll_url = job["poll_url"]
                deadline = time.monotonic() + 300
                while True:
                    status, job = request("GET", poll_url)
                    self.assertEqual(status, 200)
                    self.assertNotIn(job["status"], ("failed", "cancelled", "completed"))
                    if job.get("progress", {}).get("stage") == "sampling":
                        break
                    self.assertLess(time.monotonic(), deadline, "generation did not reach sampling")
                    time.sleep(0.5)

                start = time.monotonic()
                status, snapshot = request("GET", "/sdcpp/v1/conditioning-cache")
                self.assertEqual(status, 200)
                self.assertLess(time.monotonic() - start, 3)
                self.assertTrue(snapshot["snapshot"])
                self.assertEqual(snapshot["size"], 2)
                self.assertEqual({entry["prompt"] for entry in snapshot["entries"]},
                                 {"a blue ceramic cup on a table", "blurry"})
            finally:
                process.terminate()
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(5)


if __name__ == "__main__":
    unittest.main()
