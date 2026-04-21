"""Stage 2 - Feature engineering + split generation.

Order (matters for no-leakage):
  1. Merge GDSC2 with available cell-line IDs + drug names.
  2. Generate splits on unique COSMIC_IDs / DRUG_NAMEs.
  3. Build drug features, fit scaler on train-split drugs only.
  4. Build cell features, fit scaler on train-split cells only.
  5. Build synergy splits from DrugComb intersected with drug features.
  6. Write merged_single.parquet + merged_synergy.parquet.
  7. Assert zero leakage across LDO/LCLO/LPO partitions.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.cell_features import build_cell_features  # noqa: E402
from src.features.drug_features import build_drug_features  # noqa: E402

log = logging.getLogger(__name__)
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
SPLITS = PROC / "splits"


def _load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


# ---- Splits -------------------------------------------------------------

def _three_way_split(items: list, ratios: tuple[float, float, float], seed: int):
    rng = np.random.default_rng(seed)
    items = sorted(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)
    return set(items[:n_train]), set(items[n_train:n_train + n_val]), set(items[n_train + n_val:])


def _save_split(name: str, split_idx: dict[str, np.ndarray]) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    for part, idx in split_idx.items():
        pd.DataFrame({"row_index": idx}).to_csv(SPLITS / f"{name}_{part}.csv", index=False)


def _standard_split(df: pd.DataFrame, val: float, test: float, seed: int) -> dict[str, np.ndarray]:
    """Random 80/10/10 stratified by IC50 decile."""
    rng = np.random.default_rng(seed)
    deciles = pd.qcut(df["LN_IC50"], q=10, labels=False, duplicates="drop")
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(deciles, dropna=False):
        idx = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(val * n)
        n_test = int(test * n)
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test:n_test + n_val])
        parts["train"].append(idx[n_test + n_val:])
    return {k: np.concatenate(v) if v else np.array([], dtype=int) for k, v in parts.items()}


def _leave_out_split(df: pd.DataFrame, key_col: str, seed: int) -> dict[str, np.ndarray]:
    """70/15/15 leave-out on unique values of key_col."""
    keys = df[key_col].unique().tolist()
    tr, va, te = _three_way_split(keys, (0.70, 0.15, 0.15), seed)
    return {
        "train": df.index[df[key_col].isin(tr)].to_numpy(),
        "val":   df.index[df[key_col].isin(va)].to_numpy(),
        "test":  df.index[df[key_col].isin(te)].to_numpy(),
    }


def _assert_no_leakage(df: pd.DataFrame, key_col: str, split_idx: dict[str, np.ndarray], name: str) -> None:
    train_keys = set(df.loc[split_idx["train"], key_col])
    val_keys   = set(df.loc[split_idx["val"], key_col])
    test_keys  = set(df.loc[split_idx["test"], key_col])
    if train_keys & val_keys:
        raise AssertionError(f"{name}: train-val overlap on {key_col}: {len(train_keys & val_keys)} keys")
    if train_keys & test_keys:
        raise AssertionError(f"{name}: train-test overlap on {key_col}: {len(train_keys & test_keys)} keys")
    if val_keys & test_keys:
        raise AssertionError(f"{name}: val-test overlap on {key_col}: {len(val_keys & test_keys)} keys")


# ---- Orchestrator -------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="subsample to 2000 GDSC rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    params = _load_params()
    val_size = params["data"]["val_size"]
    test_size = params["data"]["test_size"]
    top_mut = params["data"]["top_mutation_genes"]
    radius = params["data"]["morgan_radius"]
    n_bits = params["data"]["morgan_bits"]

    log.info("load GDSC2")
    gdsc = pd.read_csv(RAW / "gdsc2.csv", usecols=["COSMIC_ID", "DRUG_NAME", "LN_IC50"])
    gdsc = gdsc.dropna(subset=["COSMIC_ID", "DRUG_NAME", "LN_IC50"]).copy()
    gdsc["COSMIC_ID"] = gdsc["COSMIC_ID"].astype(int)
    # Drop pathological IC50 outliers (validated-OK fraction in Stage 1)
    n0 = len(gdsc)
    gdsc = gdsc[(gdsc["LN_IC50"] >= -10) & (gdsc["LN_IC50"] <= 10)].copy()
    if len(gdsc) != n0:
        log.info("dropped %d GDSC rows with LN_IC50 outside [-10, 10]", n0 - len(gdsc))
    if args.debug:
        gdsc = gdsc.sample(n=min(2000, len(gdsc)), random_state=args.seed).reset_index(drop=True)
        log.info("--debug: subsampled to %d rows", len(gdsc))
    else:
        gdsc = gdsc.reset_index(drop=True)

    log.info("generate splits on GDSC2 indices (standard / LDO / LCLO)")
    std_split = _standard_split(gdsc, val_size, test_size, args.seed)
    ldo_split = _leave_out_split(gdsc, "DRUG_NAME", args.seed)
    lclo_split = _leave_out_split(gdsc, "COSMIC_ID", args.seed)

    _assert_no_leakage(gdsc, "DRUG_NAME", ldo_split, "ldo")
    _assert_no_leakage(gdsc, "COSMIC_ID", lclo_split, "lclo")

    _save_split("standard", std_split)
    _save_split("ldo", ldo_split)
    _save_split("lclo", lclo_split)

    log.info("build drug features (scaler fit on STANDARD train drugs)")
    std_train_drugs = set(gdsc.loc[std_split["train"], "DRUG_NAME"])
    drug_feats = build_drug_features(train_drug_names=std_train_drugs, radius=radius, n_bits=n_bits)

    log.info("build cell features (scaler fit on STANDARD train cells)")
    std_train_cells = set(gdsc.loc[std_split["train"], "COSMIC_ID"].astype(int))
    cell_feats = build_cell_features(train_cosmic_ids=std_train_cells, top_mutation_genes=top_mut)

    log.info("filter GDSC to rows whose cell + drug both have features")
    keep = gdsc["COSMIC_ID"].isin(cell_feats.keys()) & gdsc["DRUG_NAME"].isin(drug_feats.keys())
    # Preserve the original gdsc row index so splits (which reference the pre-merge
    # index space) still align after filtering.
    merged_single = gdsc.loc[keep].copy()
    merged_single["gdsc_row_index"] = merged_single.index.astype(int)
    merged_single = merged_single.reset_index(drop=True)
    merged_single.to_parquet(PROC / "merged_single.parquet", index=False)
    log.info("merged_single: %d rows (dropped %d without features)", len(merged_single), (~keep).sum())

    # ---- Synergy (DrugComb) --------------------------------------------
    log.info("build synergy LPO split from DrugComb")
    dc = pd.read_csv(
        RAW / "drugcomb.csv",
        usecols=["drug_row", "drug_col", "cell_line_name", "synergy_loewe"],
        low_memory=False,   # avoid pandas 3.0 _concatenate_chunks bug with usecols
    )
    dc = dc.dropna(subset=["drug_row", "drug_col", "synergy_loewe"]).copy()
    # Map cell_line_name -> COSMIC_ID via DepMap model.csv
    dm = pd.read_csv(RAW / "depmap_model.csv")
    name_col = next((c for c in ["StrippedCellLineName", "CellLineName", "cell_line_name"] if c in dm.columns), None)
    cosm_col = next((c for c in ["COSMICID", "COSMIC_ID", "cosmic_id"] if c in dm.columns), None)
    if name_col and cosm_col:
        dm_map = dm[[name_col, cosm_col]].dropna()
        dm_map[cosm_col] = pd.to_numeric(dm_map[cosm_col], errors="coerce").round().astype("Int64")
        dm_map = dm_map.dropna(subset=[cosm_col]).copy()
        dm_map[cosm_col] = dm_map[cosm_col].astype(int)
        dm_map = dm_map.drop_duplicates(subset=[name_col])
        dc = dc.merge(dm_map.rename(columns={name_col: "cell_line_name", cosm_col: "COSMIC_ID"}),
                      on="cell_line_name", how="inner")
    else:
        log.warning("DepMap model.csv has no cell_line_name column - synergy COSMIC_ID mapping skipped.")
        dc["COSMIC_ID"] = np.nan

    dc = dc[
        dc["COSMIC_ID"].isin(cell_feats.keys())
        & dc["drug_row"].isin(drug_feats.keys())
        & dc["drug_col"].isin(drug_feats.keys())
    ].reset_index(drop=True)
    log.info("synergy rows after feature-intersection: %d", len(dc))

    if len(dc) >= 30:
        dc["pair"] = dc.apply(lambda r: tuple(sorted([r["drug_row"], r["drug_col"]])), axis=1)
        lpo_split = _leave_out_split(dc, "pair", args.seed)
        _assert_no_leakage(dc, "pair", lpo_split, "lpo")
        _save_split("lpo", lpo_split)
        dc.drop(columns=["pair"]).to_parquet(PROC / "merged_synergy.parquet", index=False)
    else:
        log.warning("Too few synergy rows (%d) - writing empty merged_synergy.parquet and skipping LPO split.", len(dc))
        dc.to_parquet(PROC / "merged_synergy.parquet", index=False)

    # ---- Summary --------------------------------------------------------
    summary = {
        "standard": {k: int(len(v)) for k, v in std_split.items()},
        "ldo":      {k: int(len(v)) for k, v in ldo_split.items()},
        "lclo":     {k: int(len(v)) for k, v in lclo_split.items()},
        "n_drugs":  int(gdsc["DRUG_NAME"].nunique()),
        "n_cells":  int(gdsc["COSMIC_ID"].nunique()),
        "n_merged_single": int(len(merged_single)),
        "n_merged_synergy": int(len(dc)),
        "cell_feature_dim": int(next(iter(cell_feats.values())).shape[0]),
        "drug_feature_dim": int(next(iter(drug_feats.values())).shape[0]),
    }
    (PROC / "stage2_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Stage 2 summary: %s", summary)
    log.info("Stage 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
