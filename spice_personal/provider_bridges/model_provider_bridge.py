from __future__ import annotations

import argparse
import sys

from spice_personal.wrappers.openrouter_model import main as run_openrouter_model


MODEL_PROVIDER_OPENROUTER = "openrouter"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-personal-model-provider-bridge",
        description="Compatibility bridge for product-layer model provider wrappers.",
    )
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-key-env", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = str(args.provider).strip().lower()
    model_name = str(args.model).strip()
    api_key_env = str(args.api_key_env).strip()
    if provider != MODEL_PROVIDER_OPENROUTER:
        sys.stderr.write(
            "transport error: unsupported model provider bridge "
            f"{provider!r}; expected {MODEL_PROVIDER_OPENROUTER!r}.\n"
        )
        sys.stderr.flush()
        return 5
    return run_openrouter_model(
        [
            "--model",
            model_name,
            "--api-key-env",
            api_key_env or "OPENROUTER_API_KEY",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
