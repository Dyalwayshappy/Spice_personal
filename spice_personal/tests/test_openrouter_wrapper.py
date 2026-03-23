from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENROUTER_MODEL_ID = "anthropic/claude-opus-4-5"


class OpenRouterWrapperTests(unittest.TestCase):
    def test_wrapper_happy_path_returns_model_text(self) -> None:
        model_text = '{"result":"ok","next":"take one reversible step"}'

        with _MockOpenRouterServer(
            lambda _request: (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": model_text,
                            }
                        }
                    ]
                },
            )
        ) as server:
            completed = self._run_wrapper(
                prompt="Return a JSON object only.",
                env_overrides={
                    "OPENROUTER_API_KEY": "test-openrouter-key",
                    "SPICE_MODEL_OPENROUTER_BASE_URL": server.base_url,
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, model_text)
        self.assertEqual(completed.stderr, "")
        self.assertGreaterEqual(len(server.requests), 1)

        request = server.requests[0]
        self.assertEqual(request.get("path"), "/chat/completions")
        headers = request.get("headers", {})
        self.assertEqual(
            headers.get("Authorization"),
            "Bearer test-openrouter-key",
        )
        payload = request.get("payload", {})
        self.assertEqual(payload.get("model"), OPENROUTER_MODEL_ID)

    def test_wrapper_fails_when_api_key_is_missing(self) -> None:
        completed = self._run_wrapper(
            prompt="hello",
            env_overrides={
                "OPENROUTER_API_KEY": "",
            },
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("authentication error", completed.stderr.lower())
        self.assertIn("OPENROUTER_API_KEY", completed.stderr)

    def test_wrapper_fails_on_transport_runtime_error(self) -> None:
        completed = self._run_wrapper(
            prompt="hello",
            env_overrides={
                "OPENROUTER_API_KEY": "test-openrouter-key",
                "SPICE_MODEL_OPENROUTER_BASE_URL": self._unused_local_base_url(),
            },
        )

        self.assertEqual(completed.returncode, 5)
        self.assertIn("transport error", completed.stderr.lower())

    def test_wrapper_fails_on_invalid_or_empty_response_content(self) -> None:
        test_cases: list[dict[str, Any]] = [
            {
                "name": "missing_choices",
                "response": {"id": "missing-choices"},
                "expected": "invalid response",
            },
            {
                "name": "empty_choice_content",
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "   ",
                            }
                        }
                    ]
                },
                "expected": "empty response content",
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                with _MockOpenRouterServer(
                    lambda _request, payload=case["response"]: (200, payload)
                ) as server:
                    completed = self._run_wrapper(
                        prompt="hello",
                        env_overrides={
                            "OPENROUTER_API_KEY": "test-openrouter-key",
                            "SPICE_MODEL_OPENROUTER_BASE_URL": server.base_url,
                        },
                    )
                self.assertEqual(completed.returncode, 3)
                self.assertIn(str(case["expected"]).lower(), completed.stderr.lower())

    def test_personal_ask_uses_configured_openrouter_wrapper_with_mocked_http(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            with _MockOpenRouterServer(_personal_ask_openrouter_responder) as server:
                config_payload = {
                    "schema_version": "spice_personal.connection.v1",
                    "model": {
                        "provider": "openrouter",
                        "model": OPENROUTER_MODEL_ID,
                        "api_key_env": "OPENROUTER_API_KEY",
                        "base_url": server.base_url,
                    },
                }
                config_path = workspace / "personal.config.json"
                config_path.write_text(
                    json.dumps(config_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                completed = self._run_personal(
                    "ask",
                    "What should I do next?",
                    "--workspace",
                    str(workspace),
                    "--verbose",
                    env_overrides={
                        "OPENROUTER_API_KEY": "test-openrouter-key",
                    },
                )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("action=personal.assistant.ask_clarify", completed.stdout)
            self.assertIn(
                "suggestion=Clarifying Questions",
                completed.stdout,
            )
            self.assertGreaterEqual(len(server.requests), 2)
            for request in server.requests:
                payload = request.get("payload", {})
                self.assertEqual(payload.get("model"), OPENROUTER_MODEL_ID)

    @staticmethod
    def _run_wrapper(
        *,
        prompt: str,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("SPICE_MODEL_OPENROUTER_BASE_URL", None)
        if isinstance(env_overrides, dict):
            env.update(env_overrides)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "spice_personal.wrappers.openrouter_model",
                "--model",
                OPENROUTER_MODEL_ID,
                "--api-key-env",
                "OPENROUTER_API_KEY",
            ],
            cwd=REPO_ROOT,
            text=True,
            input=prompt,
            env=env,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _run_personal(
        *args: str,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "SPICE_PERSONAL_MODEL",
            "SPICE_PERSONAL_EXECUTOR",
            "SPICE_PERSONAL_EXECUTOR_TIMEOUT",
            "SPICE_PERSONAL_CLI_COMMAND",
            "SPICE_PERSONAL_CLI_PROFILE",
            "SPICE_PERSONAL_CLI_PROFILE_PATH",
            "SPICE_PERSONAL_CLI_PARSER_MODE",
            "SPICE_PERSONAL_SDEP_COMMAND",
        ):
            env.pop(key, None)
        if isinstance(env_overrides, dict):
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, "-m", "spice_personal.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _unused_local_base_url() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            host, port = sock.getsockname()
        finally:
            sock.close()
        return f"http://{host}:{port}"


def _personal_ask_openrouter_responder(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    payload = request.get("payload", {})
    messages = payload.get("messages")
    prompt = ""
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            prompt = str(first.get("content", ""))

    if "Decision proposals" in prompt:
        content = json.dumps(
            [
                {
                    "decision_type": "personal.assistant.llm",
                    "status": "proposed",
                    "selected_action": "personal.assistant.ask_clarify",
                    "attributes": {
                        "confidence": 0.86,
                        "urgency": "normal",
                    },
                }
            ],
            ensure_ascii=True,
        )
    elif "simulation advice" in prompt:
        content = json.dumps(
            {
                "score": 0.93,
                "confidence": 0.86,
                "urgency": "normal",
                "suggestion_text": "OpenRouter advisory: clarify your top tradeoff, then take one reversible step.",
                "simulation_rationale": "openrouter_mocked_http",
                "clarifying_questions": [
                    {
                        "question": "What is your top non-negotiable for this decision?",
                        "why": "It can change which option should be ranked first.",
                    },
                    {
                        "question": "How much short-term downside can you tolerate this year?",
                        "why": "It can reorder high-variance and stable options.",
                    },
                    {
                        "question": "What measurable outcome defines success in 3 years?",
                        "why": "It can shift recommendation toward long-term goal fit.",
                    },
                ],
            },
            ensure_ascii=True,
        )
    else:
        content = json.dumps({"score": 0.0}, ensure_ascii=True)

    return (
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        },
    )


class _MockOpenRouterServer:
    def __init__(
        self,
        responder: Callable[[dict[str, Any]], tuple[int, Any]],
    ) -> None:
        self._responder = responder
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""
        self.requests: list[dict[str, Any]] = []

    def __enter__(self) -> "_MockOpenRouterServer":
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}

                request_payload = {
                    "path": self.path,
                    "headers": {key: value for key, value in self.headers.items()},
                    "payload": payload,
                }
                parent.requests.append(request_payload)

                status, body = parent._responder(request_payload)
                if isinstance(body, (dict, list)):
                    body_bytes = json.dumps(body, ensure_ascii=True).encode("utf-8")
                    content_type = "application/json"
                else:
                    body_bytes = str(body).encode("utf-8")
                    content_type = "text/plain; charset=utf-8"

                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
