from __future__ import annotations

import argparse
import json
import os
import sys
from uuid import uuid4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-personal-cli-agent-provider-bridge",
        description="Product-layer CLI bridge for provider-style agent config.",
    )
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--auth-env", type=str, default="")
    parser.add_argument("--endpoint", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = str(args.provider).strip().lower()
    auth_env = str(args.auth_env).strip()
    endpoint = str(args.endpoint).strip()

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    action_type = str(payload.get("action_type", "")).strip()
    auth_present = bool(auth_env and os.environ.get(auth_env))
    response = {
        "outcome_type": "observation",
        "summary": f"{provider} CLI provider bridge execution",
        "execution_id": f"cli-exec-{uuid4().hex}",
        "output": {
            "provider": provider,
            "action_type": action_type,
            "auth_env": auth_env,
            "auth_present": auth_present,
            "endpoint": endpoint,
            "note": "provider bridge stub execution",
        },
    }
    sys.stdout.write(json.dumps(response, ensure_ascii=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
