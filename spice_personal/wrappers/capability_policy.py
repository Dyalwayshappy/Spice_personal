from __future__ import annotations


SUPPORT_LEVEL_FULL = "full"
SUPPORT_LEVEL_LIMITED = "limited"
SUPPORT_LEVEL_UNSPECIFIED = "unspecified"

PHASE1_P0_CAPABILITY_SUPPORT_LEVELS = {
    "personal.gather_evidence": SUPPORT_LEVEL_FULL,
    "personal.system": SUPPORT_LEVEL_FULL,
    "personal.communicate": SUPPORT_LEVEL_LIMITED,
    "personal.schedule": SUPPORT_LEVEL_LIMITED,
}


def capability_support_level(action_type: str) -> str:
    token = str(action_type or "").strip()
    if not token:
        return SUPPORT_LEVEL_UNSPECIFIED
    return PHASE1_P0_CAPABILITY_SUPPORT_LEVELS.get(token, SUPPORT_LEVEL_UNSPECIFIED)
