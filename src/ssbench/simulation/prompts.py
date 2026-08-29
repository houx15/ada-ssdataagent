"""Prompt builders, ported verbatim from SSDataBench's generation_cs.py / generation.py."""

from __future__ import annotations

from typing import Any, Mapping

from ssbench.datasets.schema import DatasetSpec
from ssbench.simulation.parsing import stringify_allowed


def _desc_allowed_blocks(spec: DatasetSpec, fields: list[str]) -> tuple[str, str]:
    desc_lines, allowed_lines = [], []
    for k in fields:
        var = spec.output_variables[k]
        allowed = var.get("allowed")
        desc_lines.append(f"- {k} ({(var.get('type') or 'static')}): {var.get('description', '')}")
        allowed_lines.append(f"- {k}: {stringify_allowed(allowed)}")
    return "\n".join(desc_lines), "\n".join(allowed_lines)


def _input_context(sampled_inputs: Mapping[str, Any]) -> str:
    if not sampled_inputs:
        return ""
    ctx_parts = [f"{k}: {v}" for k, v in sampled_inputs.items()]
    return "Conditioned on the following input attributes:\n" + ", ".join(ctx_parts) + "\n"


def build_crosssectional_prompt(spec: DatasetSpec, sampled_inputs: Mapping[str, Any]) -> str:
    """Port of generation_cs.py build_prompt (GSS-style snapshot)."""
    fields = spec.output_names
    desc_block, allowed_block = _desc_allowed_blocks(spec, fields)
    input_context = _input_context(sampled_inputs)
    static_fields = spec.static_outputs

    return f"""
You are simulating one *randomly selected* synthetic individual sampled from {spec.context}
The data represent a cross-sectional snapshot collected in the year {spec.reference_year}.
{input_context}
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
{", ".join(static_fields)}.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
{desc_block}

Allowed values:
{allowed_block}
""".strip()


def build_longitudinal_prompt(spec: DatasetSpec, sampled_inputs: Mapping[str, Any]) -> str:
    """Port of generation.py build_prompt (life_trajectory datasets such as CFPS)."""
    fields = spec.output_names
    desc_block, allowed_block = _desc_allowed_blocks(spec, fields)
    input_context = _input_context(sampled_inputs)

    age_min, age_max = spec.age_range
    sequential_fields = spec.sequential_outputs
    seq_vars_list = ", ".join(sequential_fields)

    def make_example(age: int) -> str:
        inner = ", ".join([f'"{v}": "..."' for v in sequential_fields])
        return f"    {age}: {{ {inner} }}"

    example_lines = [
        make_example(age_min),
        make_example(age_min + 1),
        "    ...",
        make_example(age_max),
    ]
    seq_example = "{\n  \"life_trajectory\": {\n" + ",\n".join(example_lines) + "\n  }\n}"

    seq_instruction = (
        f"For sequential variables ({seq_vars_list}), combine them into a single field named 'life_trajectory'. "
        f"'life_trajectory' must be a JSON object mapping each age from {age_min} to {age_max} "
        f"to a sub-object that specifies all sequential variables at that age.\n"
        f"For example:\n{seq_example}"
        f"Ensure life trajectory is aligned with static variables."
    )

    static_fields = spec.static_outputs

    return f"""
You are simulating one *randomly selected* synthetic individual sampled from {spec.context}
{input_context}
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
one key 'life_trajectory' plus all static variables {", ".join(static_fields)} .

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.
3) {seq_instruction}

Descriptions:
{desc_block}

Allowed values:
{allowed_block}
""".strip()


def build_prompt(spec: DatasetSpec, sampled_inputs: Mapping[str, Any]) -> str:
    if spec.sequential_outputs:
        return build_longitudinal_prompt(spec, sampled_inputs)
    return build_crosssectional_prompt(spec, sampled_inputs)
