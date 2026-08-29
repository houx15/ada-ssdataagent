from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from ssbench.llm.usage import BudgetExceededError, assert_budget_available, ledger_total, usage_cost


class UsageTests(unittest.TestCase):
    def test_provider_cost_wins(self):
        self.assertEqual(usage_cost({"cost": 1.25, "prompt_tokens": 99}), 1.25)

    def test_token_price_fallback(self):
        with patch.dict(os.environ, {
            "SSBENCH_INPUT_USD_PER_M": "2",
            "SSBENCH_OUTPUT_USD_PER_M": "10",
        }, clear=False):
            self.assertAlmostEqual(
                usage_cost({"prompt_tokens": 1_000_000, "completion_tokens": 500_000}),
                7.0,
            )

    def test_budget_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = os.path.join(temp, "ledger.jsonl")
            with open(ledger, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"cost_usd": 2.5}) + "\n")
            self.assertEqual(ledger_total(ledger), 2.5)
            with patch.dict(os.environ, {
                "SSBENCH_COST_LEDGER": ledger,
                "SSBENCH_BUDGET_USD": "2.5",
            }, clear=False):
                with self.assertRaises(BudgetExceededError):
                    assert_budget_available()


if __name__ == "__main__":
    unittest.main()
