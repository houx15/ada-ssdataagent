"""Typed view over a dataset simulation spec (ported from SSDataBench content yamls)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSpec:
    name: str
    context: str
    population_context: dict[str, Any]
    input_variables: dict[str, dict]
    output_variables: dict[str, dict]
    postprocess_modules: list[str] = field(default_factory=list)
    preprocessing: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, name: str, raw: dict) -> "DatasetSpec":
        return cls(
            name=name,
            context=(raw.get("context") or "").strip(),
            population_context=raw.get("population_context", {}),
            input_variables=raw.get("input_variables", {}),
            output_variables=raw.get("output_variables", {}),
            postprocess_modules=raw.get("postprocess_modules", []),
            preprocessing=raw.get("preprocessing", {}),
        )

    @property
    def input_names(self) -> list[str]:
        return list(self.input_variables)

    @property
    def output_names(self) -> list[str]:
        return list(self.output_variables)

    @property
    def sequential_outputs(self) -> list[str]:
        return [
            k for k, v in self.output_variables.items()
            if (v.get("type") or "static") == "sequential"
        ]

    @property
    def static_outputs(self) -> list[str]:
        return [
            k for k, v in self.output_variables.items()
            if (v.get("type") or "static") != "sequential"
        ]

    @property
    def age_range(self) -> tuple[int, int]:
        rng = self.population_context.get("age_range", [14, 60])
        return int(rng[0]), int(rng[-1])

    @property
    def reference_year(self):
        return self.population_context.get("reference_year")

    def allowed(self, var: str) -> Any:
        if var in self.output_variables:
            return self.output_variables[var].get("allowed")
        return self.input_variables.get(var, {}).get("allowed")
