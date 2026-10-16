"""Seed, dataset, leakage and single-record validation."""
import re

import pandas as pd

from src.config import FEATURES, FORBIDDEN_FEATURES, RANGES, TARGET_CLF, TARGET_REG


def validate_seed(student_id: str, seed: int) -> int:
    """Reject any seed that is not the last four digits of the registration number."""
    digits = re.sub(r"\D", "", student_id.split("@")[0])
    if len(digits) < 4:
        raise ValueError("Student ID must contain at least four digits (e.g. ug2424671@ines.ac.rw).")
    expected = int(digits[-4:])
    if seed != expected:
        raise ValueError(f"Invalid seed {seed}: the seed must be LLLL = {expected} "
                         f"(last four digits of '{student_id}').")
    return seed


def validate_dataset(df: pd.DataFrame) -> dict:
    """Check schema, ranges, categories and targets; return a data-quality report."""
    required = FEATURES + [TARGET_REG, TARGET_CLF]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    for col, (lo, hi) in RANGES.items():
        s = df[col].dropna()
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            raise ValueError(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df["season"].dropna().unique()) <= {"A", "B"}:
        raise ValueError("season must contain only 'A' or 'B'")
    if not set(df["cooperative_support"].dropna().unique()) <= {0, 1}:
        raise ValueError("cooperative_support must be 0 or 1")
    if df[[TARGET_REG, TARGET_CLF]].isna().any().any():
        raise ValueError("Target columns must not contain missing values")
    return {
        "rows_raw": int(len(df)),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def assert_no_leakage(feature_columns) -> None:
    """Stop if any target-derived column is used as an input feature."""
    leaked = FORBIDDEN_FEATURES & set(feature_columns)
    if leaked:
        raise ValueError(f"Target leakage: {sorted(leaked)} must not be used as features")


def validate_record(record: dict) -> list:
    """Validate one prediction record. Returns a list of error messages (empty = valid)."""
    errors = []
    if not isinstance(record, dict):
        return ["Input must be a JSON object"]
    for f in FEATURES:
        if f not in record:
            errors.append(f"Missing field: {f}")
    for f in record:
        if f not in FEATURES:
            errors.append(f"Unknown field: {f}")
    for f, (lo, hi) in RANGES.items():
        if f in record:
            v = record[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"{f} must be a number, got {type(v).__name__}")
            elif not lo <= v <= hi:
                errors.append(f"{f}={v} out of range [{lo}, {hi}]")
    if "season" in record and record["season"] not in ("A", "B"):
        errors.append("season must be 'A' or 'B'")
    if "cooperative_support" in record and record["cooperative_support"] not in (0, 1):
        errors.append("cooperative_support must be 0 or 1")
    return errors
