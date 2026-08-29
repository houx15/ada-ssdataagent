"""Schema, cross-field, and longitudinal checks for CFPS synthetic tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ssbench.datasets.schema import DatasetSpec
from ssbench.simulation.lifecycle import POSTPROCESSORS


@dataclass
class CheckResult:
    section: str
    rule: str
    eligible: int
    violations: int
    rate: float | None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _result(section: str, rule: str, mask: Iterable[bool], eligible=None,
            detail: str = "") -> CheckResult:
    mask = pd.Series(mask).fillna(False).astype(bool)
    if eligible is None:
        eligible_mask = pd.Series(True, index=mask.index)
    else:
        eligible_mask = pd.Series(eligible, index=mask.index).fillna(False).astype(bool)
    n = int(eligible_mask.sum())
    bad = int((mask & eligible_mask).sum())
    return CheckResult(section, rule, n, bad, bad / n if n else None, detail)


def _age_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[-1]))


def schema_checks(df: pd.DataFrame, spec: DatasetSpec) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for name, cfg in {**spec.input_variables, **spec.output_variables}.items():
        names = _age_columns(df, f"{name}_") if cfg.get("type") == "sequential" else [name]
        allowed = cfg.get("allowed")
        for col in names:
            if col not in df:
                checks.append(CheckResult("schema", f"{col}:present", len(df), len(df), 1.0))
                continue
            s = df[col]
            present = s.notna()
            if isinstance(allowed, list):
                checks.append(_result("schema", f"{col}:allowed",
                                      ~s.isin(allowed), present))
            elif isinstance(allowed, dict):
                numeric = pd.to_numeric(s, errors="coerce")
                special = {str(v) for v in allowed.get("special", [])}
                valid_special = s.astype(str).isin(special) if special else pd.Series(False, index=s.index)
                invalid = present & numeric.isna() & ~valid_special
                lo, hi = allowed.get("min"), allowed.get("max")
                if lo is not None:
                    invalid |= (numeric < float(lo)) & ~valid_special
                if hi is not None:
                    invalid |= (numeric > float(hi)) & ~valid_special
                checks.append(_result("schema", f"{col}:range", invalid, present,
                                      f"min={lo}, max={hi}"))
    integer_fields = {"children_number", "math_cognitive", "verbal_cognitive"}
    for field in integer_fields:
        cols = _age_columns(df, f"{field}_") if field == "children_number" else [field]
        for col in cols:
            if col not in df:
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            present = x.notna()
            checks.append(_result("schema", f"{col}:integer_grid",
                                  (x - x.round()).abs() > 1e-9, present))
    return checks


def cross_field_checks(df: pd.DataFrame) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if {"highest_education", "age_finished_education"} <= set(df):
        age = pd.to_numeric(df["age_finished_education"], errors="coerce")
        edu = df["highest_education"]
        minimum = edu.map({
            "primary school or below": 6,
            "middle school": 12,
            "high school": 15,
            "college and above": 18,
        })
        eligible = age.notna() & minimum.notna()
        checks.append(_result("cross_field", "education_completion_min_age",
                              age < minimum, eligible))
    if {"child_number", "age_at_first_child"} <= set(df):
        n = pd.to_numeric(df["child_number"], errors="coerce")
        age = pd.to_numeric(df["age_at_first_child"], errors="coerce")
        eligible = n.notna()
        checks.append(_result("cross_field", "child_count_first_child_agreement",
                              ((n <= 0) & age.notna()) | ((n > 0) & age.isna()), eligible))
    if "ever_divorced" in df:
        marital = _age_columns(df, "marital_status_")
        if marital:
            observed = df[marital].astype(str).apply(
                lambda row: row.str.lower().isin({"divorced", "widowed", "divorced or widowed"}).any(),
                axis=1,
            )
            declared = df["ever_divorced"].eq("ever divorced")
            checks.append(_result("cross_field", "divorce_summary_trajectory_agreement",
                                  observed != declared, df["ever_divorced"].notna()))
    return checks


def longitudinal_checks(df: pd.DataFrame, spec: DatasetSpec) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for name in spec.postprocess_modules:
        if name not in df or name not in POSTPROCESSORS:
            continue
        derived = POSTPROCESSORS[name](df)
        stored = df[name]
        eligible = derived.notna() | stored.notna()
        if pd.api.types.is_numeric_dtype(derived):
            a = pd.to_numeric(derived, errors="coerce")
            b = pd.to_numeric(stored, errors="coerce")
            mismatch = (a.isna() != b.isna()) | ((a - b).abs() > 1e-8)
        else:
            mismatch = derived.fillna("<NA>").astype(str) != stored.fillna("<NA>").astype(str)
        checks.append(_result("longitudinal", f"{name}:summary_matches_trajectory",
                              mismatch, eligible))

    edu_cols = _age_columns(df, "education_")
    if edu_cols:
        order = {"primary school or below": 0, "middle school": 1,
                 "high school": 2, "college and above": 3}
        values = df[edu_cols].apply(
            lambda col: pd.to_numeric(col.map(order), errors="coerce")
        ).to_numpy()
        bad = np.array([np.any(np.diff(row[np.isfinite(row)]) < 0) for row in values])
        checks.append(_result("longitudinal", "education_nondecreasing", bad))

    child_cols = _age_columns(df, "children_number_")
    if child_cols:
        values = df[child_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        bad = np.array([np.any(np.diff(row[np.isfinite(row)]) < 0) for row in values])
        checks.append(_result("longitudinal", "children_nondecreasing", bad))

    emp_cols = _age_columns(df, "employment_")
    for emp_col in emp_cols:
        age = emp_col.rsplit("_", 1)[-1]
        occ_col, income_col = f"occupation_{age}", f"income_{age}"
        if occ_col not in df or income_col not in df:
            continue
        emp = df[emp_col]
        occ = df[occ_col]
        inc = pd.to_numeric(df[income_col], errors="coerce")
        bad = ((emp == "employed") & (occ == "unemployed")) | (
            (emp == "unemployed") & (occ != "unemployed") & occ.notna()
        ) | ((emp == "unemployed") & (inc > 0))
        checks.append(_result("longitudinal", f"employment_occupation_income_{age}",
                              bad, emp.notna()))
    return checks


def validate_table(df: pd.DataFrame, spec: DatasetSpec) -> list[CheckResult]:
    return schema_checks(df, spec) + cross_field_checks(df) + longitudinal_checks(df, spec)
