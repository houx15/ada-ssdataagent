from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ssbench.llm.client import LLMClient


class LLMClientTests(unittest.TestCase):
    def make_client(self, create) -> LLMClient:
        client = LLMClient(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="test/model",
            max_retries=1,
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client

    def test_can_omit_sampling_parameters(self):
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok": true}'),
                finish_reason="stop",
            )],
            usage=None,
            model="test/model",
            id="response-1",
        ))
        client = self.make_client(create)
        with patch.dict(os.environ, {"SSBENCH_LLM_OMIT_SAMPLING_PARAMS": "1"}):
            response = client.chat("system", "user")
        self.assertTrue(response.ok)
        kwargs = create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertEqual(kwargs["max_tokens"], 32768)

    def test_terminal_api_error_is_auditable(self):
        class BadRequest(Exception):
            status_code = 400

        client = self.make_client(Mock(side_effect=BadRequest(
            "temperature is not supported"
        )))
        response = client.chat("system", "user")
        self.assertFalse(response.ok)
        self.assertEqual(response.attempts, 1)
        self.assertIn("temperature is not supported", response.error or "")


if __name__ == "__main__":
    unittest.main()
