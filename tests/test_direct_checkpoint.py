from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ssbench.simulation.methods.direct import _record_is_complete


def _spec():
    return SimpleNamespace(
        input_names=["input"],
        static_outputs=["static"],
        postprocess_modules=["derived"],
    )


def test_checkpoint_rejects_missing_postprocess_value():
    record = {"profile_id": 1, "input": "x", "static": "ok"}
    with patch(
        "ssbench.simulation.methods.direct.apply_postprocess",
        side_effect=lambda frame, modules: frame.assign(derived=np.nan),
    ):
        assert not _record_is_complete(record, _spec())


def test_checkpoint_accepts_complete_postprocess_value():
    record = {"profile_id": 1, "input": "x", "static": "ok"}
    with patch(
        "ssbench.simulation.methods.direct.apply_postprocess",
        side_effect=lambda frame, modules: frame.assign(derived=3),
    ):
        assert _record_is_complete(record, _spec())
