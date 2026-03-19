from __future__ import annotations

import unittest

from spice.llm.core.provider import LLMAuthError, LLMResponseError
from spice_personal.wrappers.errors import (
    WrapperErrorCategory,
    format_wrapper_error,
    model_unsupported_capability_error,
    wrap_model_exception,
)


class WrapperErrorTaxonomyTests(unittest.TestCase):
    def test_wrap_model_exception_classifies_auth_config(self) -> None:
        wrapped = wrap_model_exception(
            LLMAuthError("missing api key"),
            stage="decision_propose",
        )
        self.assertEqual(wrapped.info.category, WrapperErrorCategory.AUTH_CONFIG)
        self.assertEqual(wrapped.info.code, "model.auth_config")
        self.assertEqual(wrapped.info.stage, "decision_propose")

    def test_wrap_model_exception_classifies_response_validity(self) -> None:
        wrapped = wrap_model_exception(
            LLMResponseError("No JSON payload found in response."),
            stage="decision_propose",
        )
        self.assertEqual(wrapped.info.category, WrapperErrorCategory.RESPONSE_VALIDITY)
        self.assertEqual(wrapped.info.code, "model.response_validity")

    def test_format_wrapper_error_includes_category_and_code(self) -> None:
        wrapped = model_unsupported_capability_error(
            "unsupported action",
            stage="decision_propose",
        )
        rendered = format_wrapper_error(wrapped)
        self.assertIn("category=unsupported_capability", rendered)
        self.assertIn("code=model.unsupported_capability", rendered)


if __name__ == "__main__":
    unittest.main()
