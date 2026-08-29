from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


class FutureExperimentConfigTests(unittest.TestCase):
    def test_yaml_date_can_be_snapshotted(self):
        path = Path(__file__).parents[1] / "configs" / "experiments" / "future_p0.yaml"
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        payload = json.dumps({"config": cfg}, default=str)
        self.assertIn("2026-08-29", payload)


if __name__ == "__main__":
    unittest.main()
