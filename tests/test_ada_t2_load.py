from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ada_t2_load import stringified_value_counts  # noqa: E402


def test_stringified_value_counts_handles_mixed_types() -> None:
    values = np.array([1.0, "1", np.nan, None, 2, "two"], dtype=object)
    permuted = values[[5, 2, 0, 4, 1, 3]]

    assert stringified_value_counts(values) == stringified_value_counts(permuted)


def test_stringified_value_counts_detects_changed_marginal() -> None:
    before = np.array([1.0, "1", np.nan, None], dtype=object)
    after = np.array([1.0, "changed", np.nan, None], dtype=object)

    assert stringified_value_counts(before) != stringified_value_counts(after)
