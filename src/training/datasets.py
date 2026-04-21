"""torch.utils.data.Dataset wrappers over the Stage-2 parquet + pkl artifacts.

Two datasets:
  - SingleDrugDataset: yields (cell_vec, drug_vec, ln_ic50)
  - SynergyDataset:    yields (cell_vec, drug_a_vec, drug_b_vec, synergy_loewe)

Both accept a `mask_indices` argument: an array of row indices (into the
underlying parquet) that this dataset should expose. This is how we slice
train/val/test without copying the feature matrices.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_feature_caches() -> tuple[dict, dict]:
    with open(PROC / "cell_features.pkl", "rb") as f:
        cell = pickle.load(f)
    with open(PROC / "drug_features.pkl", "rb") as f:
        drug = pickle.load(f)
    return cell, drug


def _lookup(d: dict, key):
    v = d.get(key)
    if v is None:
        raise KeyError(f"feature cache miss: {key}")
    return v


class SingleDrugDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cell_feats: dict, drug_feats: dict):
        """df: columns [COSMIC_ID, DRUG_NAME, LN_IC50] (merged_single.parquet rows)."""
        self.cosm = df["COSMIC_ID"].to_numpy()
        self.drug = df["DRUG_NAME"].to_numpy()
        self.y = df["LN_IC50"].to_numpy(dtype=np.float32)
        self.cell_feats = cell_feats
        self.drug_feats = drug_feats

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        c = _lookup(self.cell_feats, int(self.cosm[i]))
        d = _lookup(self.drug_feats, str(self.drug[i]))
        return (
            torch.from_numpy(c),
            torch.from_numpy(d),
            torch.tensor(self.y[i], dtype=torch.float32),
        )


class SynergyDataset(Dataset):
    SYNERGY_CLIP = 100.0  # Loewe scores beyond this are drug-pair pathologies, not signal

    def __init__(self, df: pd.DataFrame, cell_feats: dict, drug_feats: dict):
        """df: columns [drug_row, drug_col, COSMIC_ID, synergy_loewe].
        DrugComb's synergy_loewe field has ~0.1% NaN rows and a long tail
        (-124 to +417). We drop NaN rows and clip to [-SYNERGY_CLIP, SYNERGY_CLIP];
        an unclipped outlier turns the multitask gradient into NaN via the shared
        encoders and kills single-drug training too.
        """
        y = pd.to_numeric(df["synergy_loewe"], errors="coerce")
        mask = y.notna()
        df = df.loc[mask].reset_index(drop=True)
        y  = y.loc[mask].reset_index(drop=True)
        y  = y.clip(lower=-self.SYNERGY_CLIP, upper=self.SYNERGY_CLIP)
        self.cosm  = df["COSMIC_ID"].to_numpy()
        self.a     = df["drug_row"].to_numpy()
        self.b     = df["drug_col"].to_numpy()
        self.y     = y.to_numpy(dtype=np.float32)
        self.cell_feats = cell_feats
        self.drug_feats = drug_feats

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        c = _lookup(self.cell_feats, int(self.cosm[i]))
        a = _lookup(self.drug_feats, str(self.a[i]))
        b = _lookup(self.drug_feats, str(self.b[i]))
        return (
            torch.from_numpy(c),
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.tensor(self.y[i], dtype=torch.float32),
        )


def load_single_split(split_name: str, part: str,
                      cell_feats: dict, drug_feats: dict) -> SingleDrugDataset:
    """split_name in {standard, ldo, lclo}; part in {train, val, test}."""
    df = pd.read_parquet(PROC / "merged_single.parquet")
    idx = pd.read_csv(PROC / "splits" / f"{split_name}_{part}.csv")["row_index"].to_numpy()
    mask = df["gdsc_row_index"].isin(idx)
    sub = df.loc[mask].reset_index(drop=True)
    return SingleDrugDataset(sub, cell_feats, drug_feats)


def load_synergy_split(part: str, cell_feats: dict, drug_feats: dict) -> SynergyDataset | None:
    """LPO split on merged_synergy.parquet. Returns None if split file missing
    (e.g., too few synergy rows after feature intersection)."""
    df = pd.read_parquet(PROC / "merged_synergy.parquet")
    if len(df) == 0:
        return None
    split_path = PROC / "splits" / f"lpo_{part}.csv"
    if not split_path.exists():
        return None
    idx = pd.read_csv(split_path)["row_index"].to_numpy()
    sub = df.iloc[idx].reset_index(drop=True) if len(idx) else df.iloc[:0]
    if len(sub) == 0:
        return None
    return SynergyDataset(sub, cell_feats, drug_feats)
