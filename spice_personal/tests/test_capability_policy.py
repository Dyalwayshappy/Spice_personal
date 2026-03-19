from __future__ import annotations

import unittest

from spice_personal.wrappers.capability_policy import (
    SUPPORT_LEVEL_FULL,
    SUPPORT_LEVEL_LIMITED,
    SUPPORT_LEVEL_UNSPECIFIED,
    capability_support_level,
)


class CapabilityPolicyTests(unittest.TestCase):
    def test_phase1_p0_support_levels_are_explicit(self) -> None:
        self.assertEqual(
            capability_support_level("personal.gather_evidence"),
            SUPPORT_LEVEL_FULL,
        )
        self.assertEqual(
            capability_support_level("personal.system"),
            SUPPORT_LEVEL_FULL,
        )
        self.assertEqual(
            capability_support_level("personal.communicate"),
            SUPPORT_LEVEL_LIMITED,
        )
        self.assertEqual(
            capability_support_level("personal.schedule"),
            SUPPORT_LEVEL_LIMITED,
        )

    def test_unknown_capability_defaults_to_unspecified(self) -> None:
        self.assertEqual(
            capability_support_level("personal.unknown"),
            SUPPORT_LEVEL_UNSPECIFIED,
        )


if __name__ == "__main__":
    unittest.main()
