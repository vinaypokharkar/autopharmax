"""Drug feature vectors: 2048 Morgan bits + ~180-200 RDKit descriptors.

Scaler + median-imputer are fit on TRAIN drug names only. Drugs where
Chem.MolFromSmiles() returns None are dropped - never imputed.
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def _compute_raw(smiles_df: pd.DataFrame, radius: int, n_bits: int):
    """Returns (good_names, morgan_mat, desc_mat, desc_names, dropped_names)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdFingerprintGenerator

    mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    desc_list = Descriptors.descList
    desc_names = [d[0] for d in desc_list]
    desc_fns = [d[1] for d in desc_list]

    names, morgans, descs, dropped = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in smiles_df.iterrows():
            name = row["drug_name"]
            smi = row["smiles"]
            if not isinstance(smi, str) or not smi:
                dropped.append(name); continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                dropped.append(name); continue
            try:
                fp = mgen.GetFingerprint(mol)
                morgan = np.array(fp, dtype=np.uint8)
                vals = np.array([fn(mol) for fn in desc_fns], dtype=np.float64)
            except Exception as e:
                log.warning("rdkit error for %s: %s", name, e)
                dropped.append(name); continue
            names.append(name)
            morgans.append(morgan)
            descs.append(vals)

    if not names:
        raise AssertionError("no drugs produced valid features")

    morgan_mat = np.vstack(morgans)
    desc_mat = np.vstack(descs)
    return names, morgan_mat, desc_mat, desc_names, dropped


def _clean_descriptors(desc_mat: np.ndarray, desc_names: list[str], drop_nan_thresh: float = 0.20):
    """Drop descriptor columns with >20% NaN/Inf. Returns (cleaned_mat, kept_names)."""
    n = desc_mat.shape[0]
    bad = ~np.isfinite(desc_mat)
    bad_frac = bad.mean(axis=0)
    keep = bad_frac <= drop_nan_thresh
    kept_names = [n_ for n_, k in zip(desc_names, keep) if k]
    cleaned = desc_mat[:, keep].astype(np.float64)
    cleaned[~np.isfinite(cleaned)] = np.nan
    log.info("Descriptors kept: %d / %d (dropped cols with >%.0f%% NaN/Inf)",
             len(kept_names), len(desc_names), drop_nan_thresh * 100)
    return cleaned, kept_names


def build_drug_features(train_drug_names: set[str] | None = None,
                        radius: int = 2, n_bits: int = 2048) -> dict:
    smiles_df = pd.read_csv(RAW / "drug_smiles.csv")
    names, morgan_mat, desc_mat, desc_names_all, dropped = _compute_raw(smiles_df, radius, n_bits)
    log.info("Drugs with valid molecules: %d (dropped %d)", len(names), len(dropped))

    desc_clean, desc_names = _clean_descriptors(desc_mat, desc_names_all)

    # Impute + scale on TRAIN drugs only.
    if train_drug_names is None:
        log.warning("No train_drug_names provided; imputer+scaler fit on ALL drugs.")
        train_mask = np.ones(len(names), dtype=bool)
    else:
        train_mask = np.array([n in train_drug_names for n in names])
        if train_mask.sum() < 5:
            raise AssertionError(f"only {train_mask.sum()} train drugs - too few to fit scaler")

    train_vals = desc_clean[train_mask]
    medians = np.nanmedian(train_vals, axis=0)
    # Any descriptor entirely NaN in training set -> fall back to 0
    medians = np.where(np.isnan(medians), 0.0, medians)
    # Impute everywhere (including non-train) using train medians
    desc_imputed = np.where(np.isnan(desc_clean), medians, desc_clean)

    scaler = StandardScaler()
    scaler.fit(desc_imputed[train_mask])
    desc_scaled = scaler.transform(desc_imputed).astype(np.float32)

    # Concatenate: [morgan (2048), descriptors (~190)]
    final = np.concatenate([morgan_mat.astype(np.float32), desc_scaled], axis=1)
    log.info("Drug feature matrix: %s", final.shape)

    features = {n: final[i] for i, n in enumerate(names)}

    PROC.mkdir(parents=True, exist_ok=True)
    with open(PROC / "drug_features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(PROC / "drug_desc_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "medians": medians, "desc_names": desc_names}, f)

    meta_path = PROC / "feature_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["drug_descriptors"] = desc_names
    meta["morgan_bits"] = int(n_bits)
    meta["drug_feature_dim"] = int(final.shape[1])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if dropped:
        (RAW / "smiles_failures.txt").open("a", encoding="utf-8").write(
            "\n".join(f"{d}\t(rdkit parse failed)" for d in dropped) + "\n"
        )

    return features
