"""Life-course derived variables, ported from SSDataBench's postprocess_utils.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_age_finished_education(df, educ_prefix="education_"):
    educ_cols = [c for c in df.columns if c.startswith(educ_prefix)]
    if not educ_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    educ_cols = sorted(educ_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    ages = [int(c.rsplit("_", 1)[1]) for c in educ_cols]

    def _find_last_change(row):
        levels = [row[c] for c in educ_cols]
        last_level, last_change_age = None, None
        for age, level in zip(ages, levels):
            if pd.isna(level):
                continue
            if last_level is None:
                last_level, last_change_age = level, age
            elif level != last_level:
                last_level, last_change_age = level, age
        if last_change_age is None:
            valid_ages = [a for a, l in zip(ages, levels) if pd.notna(l)]
            return valid_ages[0] if valid_ages else np.nan
        return last_change_age

    return df.apply(_find_last_change, axis=1)


def compute_age_at_first_marriage(df, marital_prefix="marital_status_"):
    marital_cols = [c for c in df.columns if c.startswith(marital_prefix)]
    if not marital_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    marital_cols = sorted(marital_cols, key=lambda c: int(c.rsplit("_", 2)[-1]))
    ages = [int(c.rsplit("_", 2)[-1]) for c in marital_cols]

    def _find_first_marriage(row):
        for age, status in zip(ages, [row[c] for c in marital_cols]):
            if pd.isna(status):
                continue
            if str(status).lower() == "married":
                return age
        return np.nan

    return df.apply(_find_first_marriage, axis=1)


def compute_ever_divorced(df, marital_prefix="marital_status_"):
    allowed = {"divorced", "widowed", "divorced or widowed"}
    marital_cols = [c for c in df.columns if c.startswith(marital_prefix)]
    if not marital_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    marital_cols = sorted(marital_cols, key=lambda c: int(c.rsplit("_", 2)[-1]))

    def _ever_divorced(row):
        for raw_status in [row[c] for c in marital_cols]:
            if pd.isna(raw_status):
                continue
            if str(raw_status).strip().lower() in allowed:
                return "ever divorced"
        return "never divorced"

    return df.apply(_ever_divorced, axis=1)


def compute_age_started_work(df, work_prefix="employment_"):
    work_cols = [c for c in df.columns if c.startswith(work_prefix)]
    if not work_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    work_cols = sorted(work_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    ages = [int(c.rsplit("_", 1)[1]) for c in work_cols]

    def _find_started_work(row):
        for age, status in zip(ages, [row[c] for c in work_cols]):
            if pd.isna(status):
                continue
            if str(status).lower() == "employed":
                return age
        return np.nan

    return df.apply(_find_started_work, axis=1)


def compute_age_at_first_child(df, child_prefix="children_number_"):
    child_cols = [c for c in df.columns if c.startswith(child_prefix)]
    if not child_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    child_cols = sorted(child_cols, key=lambda c: int(c.rsplit("_", 2)[-1]))
    ages = [int(c.rsplit("_", 2)[-1]) for c in child_cols]

    def _find_first_child(row):
        for age, raw in zip(ages, [row[c] for c in child_cols]):
            try:
                num = float(raw)
            except (TypeError, ValueError):
                continue
            if pd.isna(num):
                continue
            if num >= 1:
                return age
        return np.nan

    return df.apply(_find_first_child, axis=1)


def compute_child_number(df, child_prefix="children_number_"):
    child_cols = sorted(
        [c for c in df.columns if c.startswith(child_prefix)],
        key=lambda c: int(c.rsplit("_", 2)[-1]),
    )
    if not child_cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    def _last_valid(row):
        for c in reversed(child_cols):
            if pd.notna(row[c]):
                return row[c]
        return np.nan

    return df.apply(_last_valid, axis=1)


def compute_occupation_30_40(df, occ_prefix="occupation_"):
    occ_cols = [c for c in df.columns if c.startswith(occ_prefix)]
    if not occ_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    occ_cols = sorted(occ_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    ages = [int(c.rsplit("_", 1)[1]) for c in occ_cols]
    selected_cols = [c for age, c in zip(ages, occ_cols) if 30 <= age <= 40]
    if not selected_cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    def _mode_occupation(row):
        vals = [row[c] for c in selected_cols if pd.notna(row[c])]
        if not vals:
            return np.nan
        return pd.Series(vals).mode().iloc[0]

    return df.apply(_mode_occupation, axis=1)


def compute_mean_income_30_40(df, income_prefix="income_"):
    income_cols = [c for c in df.columns if c.startswith(income_prefix)]
    if not income_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    income_cols = sorted(income_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    ages = [int(c.rsplit("_", 1)[1]) for c in income_cols]
    selected_cols = [c for age, c in zip(ages, income_cols) if 30 <= age <= 40]
    if not selected_cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    def _mean_income(row):
        vals = [row[c] for c in selected_cols if pd.notna(row[c])]
        if not vals:
            return np.nan
        return float(np.mean([float(v) for v in vals]))

    return df.apply(_mean_income, axis=1)


def compute_highest_education(df, educ_prefix="education_"):
    educ_cols = [c for c in df.columns if c.startswith(educ_prefix)]
    if not educ_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    educ_cols = sorted(educ_cols, key=lambda c: int(c.rsplit("_", 1)[1]))

    def _last_education(row):
        for col in reversed(educ_cols):
            val = row[col]
            if pd.notna(val):
                return val
        return np.nan

    return df.apply(_last_education, axis=1)


POSTPROCESSORS = {
    "age_at_first_marriage": compute_age_at_first_marriage,
    "ever_divorced": compute_ever_divorced,
    "age_at_first_child": compute_age_at_first_child,
    "age_finished_education": compute_age_finished_education,
    "age_started_work": compute_age_started_work,
    "child_number": compute_child_number,
    "occupation_30_40": compute_occupation_30_40,
    "mean_income_30_40": compute_mean_income_30_40,
    "highest_education": compute_highest_education,
}


def apply_postprocess(df: pd.DataFrame, modules: list[str]) -> pd.DataFrame:
    """Run named derive modules in order (same names as SSDataBench postprocess_modules)."""
    for name in modules:
        func = POSTPROCESSORS.get(name)
        if func is None:
            print(f"[postprocess] unknown module '{name}', skipping")
            continue
        df[name] = func(df)
    return df
