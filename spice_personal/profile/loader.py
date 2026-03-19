from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from spice_personal.profile.contract import PROFILE_RELATIVE_PATH


def workspace_profile_path(workspace: Path) -> Path:
    return workspace / PROFILE_RELATIVE_PATH


def ensure_workspace_profile(workspace: Path, *, force: bool = False) -> Path:
    profile_path = workspace_profile_path(workspace)
    if profile_path.exists() and not force:
        return profile_path

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    asset = files("spice_personal.profile.assets").joinpath("personal.profile.v1.json")
    payload = json.loads(asset.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Built-in profile payload must be an object.")
    profile_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile_path


def load_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Profile payload must be an object: {path}")
    return dict(payload)


def profile_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
