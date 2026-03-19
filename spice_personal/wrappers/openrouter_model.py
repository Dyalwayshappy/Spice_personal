from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
BASE_URL_ENV = "SPICE_MODEL_OPENROUTER_BASE_URL"
SITE_URL_ENV = "SPICE_MODEL_OPENROUTER_SITE_URL"
APP_NAME_ENV = "SPICE_MODEL_OPENROUTER_APP_NAME"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-model-openrouter",
        description="OpenRouter subprocess wrapper for SPICE personal model integration.",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_name = str(args.model).strip()
    if not model_name:
        _write_stderr("invalid response: model id cannot be empty.\n")
        return 3
    api_key_env = str(args.api_key_env).strip() or "OPENROUTER_API_KEY"
    base_url = (
        str(args.base_url).strip()
        or str(os.environ.get(BASE_URL_ENV, "")).strip()
        or DEFAULT_BASE_URL
    )
    timeout_seconds = _resolve_timeout(args.timeout_seconds)
    prompt = sys.stdin.read()
    if not prompt.strip():
        _write_stderr("invalid response: input prompt is empty.\n")
        return 3

    api_key = str(os.environ.get(api_key_env, "")).strip()
    if not api_key:
        _write_stderr(
            f"authentication error: missing API key env {api_key_env!r}.\n"
        )
        return 2

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = str(os.environ.get(SITE_URL_ENV, "")).strip()
    app_name = str(os.environ.get(APP_NAME_ENV, "")).strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    request_url = base_url.rstrip("/") + "/chat/completions"
    try:
        response_payload = _post_json(
            request_url,
            payload=payload,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
    except urllib.error.HTTPError as exc:
        body = _read_http_error_body(exc)
        if exc.code in {401, 403}:
            _write_stderr(f"authentication error: http {exc.code}. {body}\n")
            return 2
        if exc.code == 429:
            _write_stderr(f"rate limit error: http 429. {body}\n")
            return 4
        _write_stderr(f"transport error: http {exc.code}. {body}\n")
        return 5
    except ValueError as exc:
        _write_stderr(f"invalid response: {exc}\n")
        return 3
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        _write_stderr(f"transport error: {exc}\n")
        return 5
    except Exception as exc:
        _write_stderr(f"transport error: {exc}\n")
        return 5

    try:
        output_text = _extract_output_text(response_payload)
    except ValueError as exc:
        _write_stderr(f"invalid response: {exc}\n")
        return 3

    if not output_text.strip():
        _write_stderr("invalid response: empty response content.\n")
        return 3
    sys.stdout.write(output_text)
    sys.stdout.flush()
    return 0


def _resolve_timeout(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    if parsed <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return parsed


def _post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response payload must be a JSON object")
    return parsed


def _extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing non-empty choices array")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choices[0] must be an object")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                text = entry.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("missing textual content in choices")


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8").strip()
    except Exception:
        payload = ""
    if not payload:
        return "<empty>"
    return payload


def _write_stderr(message: str) -> None:
    sys.stderr.write(message)
    sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
