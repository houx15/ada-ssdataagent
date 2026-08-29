from __future__ import annotations

import unittest

import pandas as pd

from ssbench.datasets.registry import load_dataset
from ssbench.diagnostics.consistency import cross_field_checks, longitudinal_checks


class ConsistencyTests(unittest.TestCase):
    def test_cross_field_violations_are_counted(self):
        df = pd.DataFrame({
            "highest_education": ["college and above", "middle school"],
            "age_finished_education": [12, 14],
            "child_number": [0, 2],
            "age_at_first_child": [24, None],
        })
        got = {r.rule: r for r in cross_field_checks(df)}
        self.assertEqual(got["education_completion_min_age"].violations, 1)
        self.assertEqual(got["child_count_first_child_agreement"].violations, 2)

    def test_longitudinal_monotonicity(self):
        df = pd.DataFrame({
            "education_14": ["middle school", "primary school or below"],
            "education_15": ["primary school or below", "middle school"],
            "children_number_14": [1, 0],
            "children_number_15": [0, 1],
        })
        got = {r.rule: r for r in longitudinal_checks(df, load_dataset("cfps"))}
        self.assertEqual(got["education_nondecreasing"].violations, 1)
        self.assertEqual(got["children_nondecreasing"].violations, 1)


if __name__ == "__main__":
    unittest.main()
