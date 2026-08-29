from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.future_experiments import direct_artifact_error


class FutureExperimentConfigTests(unittest.TestCase):
    def test_yaml_date_can_be_snapshotted(self):
        path = Path(__file__).parents[1] / "configs" / "experiments" / "future_p0.yaml"
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        payload = json.dumps({"config": cfg}, default=str)
        self.assertIn("2026-08-29", payload)
        self.assertEqual(cfg["providers"]["openrouter"]["request_timeout_seconds"], 120)

    def test_direct_artifact_requires_complete_population(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "meta.json").write_text(json.dumps({
                "n": 2, "n_complete": 0, "n_checkpointed": 0,
            }), encoding="utf-8")
            pd.DataFrame({"profile_id": [0, 1]}).to_csv(run_dir / "sim.csv", index=False)
            self.assertEqual(
                direct_artifact_error(run_dir, 2),
                "direct meta.n_complete=0, expected 2",
            )

    def test_direct_artifact_accepts_complete_population(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "meta.json").write_text(json.dumps({
                "n": 2, "n_complete": 2, "n_checkpointed": 2,
            }), encoding="utf-8")
            pd.DataFrame({"profile_id": [0, 1]}).to_csv(run_dir / "sim.csv", index=False)
            self.assertIsNone(direct_artifact_error(run_dir, 2))


if __name__ == "__main__":
    unittest.main()
