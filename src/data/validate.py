"""Raw data validation gate.

Uses plain pandas assertions (not Great Expectations' full fluent API) so
we stay fast and avoid GE's context directory bloat. The checks match the
GE expectations listed in the spec; each failure raises a single clear
AssertionError that stops Stage 1 before DVC-tracking corrupt inputs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def _require_columns(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{source}: missing columns {missing}")


def validate_gdsc2() -> pd.DataFrame:
    path = RAW / "gdsc2.csv"
    df = pd.read_csv(path)
    _require_columns(df, ["COSMIC_ID", "DRUG_NAME", "LN_IC50"], path.name)
    if df["DRUG_NAME"].isna().any():
        raise AssertionError("gdsc2: NaN DRUG_NAME rows present")
    bad_mask = (df["LN_IC50"] < -10) | (df["LN_IC50"] > 10)
    bad_frac = bad_mask.mean()
    if bad_frac > 0.05:
        raise AssertionError(
            f"gdsc2: {bad_mask.sum()} / {len(df)} rows ({bad_frac:.1%}) have LN_IC50 outside [-10, 10]. "
            "Source file may be corrupt.")
    if bad_mask.any():
        log.warning("gdsc2: %d rows (%.2f%%) with LN_IC50 outside [-10, 10] - will be dropped in Stage 2.",
                    int(bad_mask.sum()), bad_frac * 100)
    log.info("gdsc2 OK: %d rows (clean %d), %d drugs, %d cell lines",
             len(df), int((~bad_mask).sum()), df["DRUG_NAME"].nunique(), df["COSMIC_ID"].nunique())
    return df


def validate_smiles() -> pd.DataFrame:
    path = RAW / "drug_smiles.csv"
    df = pd.read_csv(path)
    _require_columns(df, ["drug_name", "smiles"], path.name)
    dupes = df["drug_name"].duplicated()
    if dupes.any():
        raise AssertionError(f"drug_smiles: {dupes.sum()} duplicate drug_name rows")
    if df["smiles"].isna().any():
        raise AssertionError("drug_smiles: NaN SMILES present")
    log.info("drug_smiles OK: %d drugs", len(df))
    return df


def validate_expression() -> None:
    path = RAW / "ccle_expression.csv"
    if not path.exists():
        raise AssertionError(f"missing {path}")
    # Read header + row count cheaply - file can be 200MB+.
    nrows = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
    if nrows <= 1000:
        raise AssertionError(f"ccle_expression: only {nrows} rows (expected >1000)")
    log.info("ccle_expression OK: %d rows", nrows)


def validate_drugcomb() -> pd.DataFrame:
    path = RAW / "drugcomb.csv"
    df = pd.read_csv(path)
    _require_columns(df, ["drug_row", "drug_col", "cell_line_name", "synergy_loewe"], path.name)
    log.info("drugcomb OK: %d rows", len(df))
    return df


def run_all() -> None:
    validate_gdsc2()
    validate_smiles()
    validate_expression()
    validate_drugcomb()
    log.info("All validations passed.")
