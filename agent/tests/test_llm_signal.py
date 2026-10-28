"""LLM reader wiring tests - stub endpoint in-process, no model needed."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from forecast_core import llm_signal  # noqa: E402


def _corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    text = tmp_path / "text"
    text.mkdir()
    (text / "fomc_2024_06_12.txt").write_text(
        "The committee sees upside risk and may hike.")
    return text


def _serve(reply: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def env():
    saved = dict(os.environ)
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)


def test_llm_reply_is_used_and_clamped(tmp_path, env):
    server = _serve('Sure! {"drift_in_sigmas": 4.7, '
                    '"per_asset": {"UST_2Y": -9.0, "GHOST": 0.2}, '
                    '"reasons": ["hawkish statement"]}')
    env["MODEL_ENDPOINT"] = f"http://127.0.0.1:{server.server_port}"
    env["MODEL_NAME"] = "stub"
    signal = llm_signal.read_corpus(_corpus(tmp_path), "2024-06-28",
                                    ["UST_2Y"], [21], "level")
    server.shutdown()
    assert signal["reader"].startswith("llm")
    assert signal["drift_in_sigmas"] == 0.5          # clamped from 4.7
    assert signal["per_asset_drift"] == {"UST_2Y": -0.5}  # clamped, GHOST dropped
    assert signal["llm"]["reasons"] == ["hawkish statement"]


def test_dead_endpoint_falls_back_to_lexicon(tmp_path, env):
    env["MODEL_ENDPOINT"] = "http://127.0.0.1:9"  # nothing listens
    env["MODEL_NAME"] = "stub"
    signal = llm_signal.read_corpus(_corpus(tmp_path), "2024-06-28",
                                    ["UST_2Y"], [21], "level")
    assert signal["reader"] == "lexicon (llm fallback)"
    assert "drift_in_sigmas" in signal          # lexicon numbers intact
    assert "per_asset_drift" not in signal


def test_no_endpoint_is_pure_lexicon(tmp_path, env):
    env.pop("MODEL_ENDPOINT", None)
    env.pop("MODEL_NAME", None)
    signal = llm_signal.read_corpus(_corpus(tmp_path), "2024-06-28",
                                    ["UST_2Y"], [21], "level")
    assert signal["reader"] == "lexicon (no model endpoint)"


def test_non_json_reply_falls_back(tmp_path, env):
    server = _serve("I think rates will go up, good luck!")
    env["MODEL_ENDPOINT"] = f"http://127.0.0.1:{server.server_port}"
    env["MODEL_NAME"] = "stub"
    signal = llm_signal.read_corpus(_corpus(tmp_path), "2024-06-28",
                                    ["UST_2Y"], [21], "level")
    server.shutdown()
    assert signal["reader"] == "lexicon (llm fallback)"
    assert any("not JSON" in n for n in signal["reader_notes"])
