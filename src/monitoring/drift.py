"""Feature drift detection via per-feature KS-test.

Compares the distribution of CELL feature vectors served in the last N
days (pulled from predictions.db + cell_features.pkl) against the
distribution of training cells.

If more than `drift_fraction` of features show KS p-value below
`drift_threshold`, we emit `drift_alert.json` as a sentinel for a
cron/GitHub-Actions retraining trigger. Even with no alert we always
write `drift_report_{date}.json` for trend tracking.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

log = logging.getLogger(__name__)


def _load_training_cells() -> np.ndarray:
    """Stack the 1365-dim cell feature matrix for all training cells."""
    import json as _json
    meta = _json.loads((PROC / "feature_metadata.json").read_text(encoding="utf-8"))
    cell_names = meta["cell_features"]
    with open(PROC / "cell_features.pkl", "rb") as f:
        cell_feats: dict[int, np.ndarray] = pickle.load(f)
    # Use the STANDARD train split as the reference distribution.
    std_train = pd.read_csv(PROC / "splits" / "standard_train.csv")["row_index"].to_numpy()
    single = pd.read_parquet(PROC / "merged_single.parquet")
    train_mask = single["gdsc_row_index"].isin(std_train)
    train_cosms = single.loc[train_mask, "COSMIC_ID"].astype(int).unique()
    mat = np.stack([cell_feats[int(c)] for c in train_cosms if int(c) in cell_feats])
    return mat, cell_names


def _recent_served_cells(db_path: Path, days: int) -> np.ndarray:
    if not db_path.exists():
        return np.array([])
    cutoff = time.time() - days * 86400
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT cosmic_id FROM predictions WHERE ts >= ?", (cutoff,),
        ).fetchall()
    cosmic_ids = [int(r[0]) for r in rows]
    if not cosmic_ids:
        return np.array([])
    with open(PROC / "cell_features.pkl", "rb") as f:
        cell_feats = pickle.load(f)
    vecs = [cell_feats[c] for c in cosmic_ids if c in cell_feats]
    if not vecs:
        return np.array([])
    return np.stack(vecs)


def run(
    db_path: Path = ROOT / "predictions.db",
    days: int = 7,
    out_dir: Path = ROOT,
) -> dict:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    p_threshold = float(params["monitoring"]["drift_threshold"])

    train_mat, names = _load_training_cells()
    served_mat = _recent_served_cells(db_path, days)
    log.info("training cells: %d x %d  |  served cells in last %dd: %d",
             train_mat.shape[0], train_mat.shape[1], days,
             served_mat.shape[0] if served_mat.size else 0)

    per_feature = []
    shifted = 0
    if served_mat.size and served_mat.shape[0] >= 10:
        for i, name in enumerate(names):
            try:
                stat, p = ks_2samp(train_mat[:, i], served_mat[:, i])
                p = float(p)
                if p < p_threshold:
                    shifted += 1
                per_feature.append({"feature": name, "ks_stat": float(stat), "p_value": p})
            except Exception as e:
                per_feature.append({"feature": name, "ks_stat": None, "p_value": None,
                                    "error": str(e)})
    else:
        log.warning("too few served cells (%d) - drift check skipped",
                    served_mat.shape[0] if served_mat.size else 0)

    shifted_frac = (shifted / len(names)) if names else 0.0
    alert = shifted_frac > 0.05

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": int(days),
        "n_training_cells": int(train_mat.shape[0]),
        "n_served_cells":  int(served_mat.shape[0]) if served_mat.size else 0,
        "n_features":       int(len(names)),
        "n_shifted":        int(shifted),
        "shifted_fraction": float(shifted_frac),
        "p_threshold":      float(p_threshold),
        "alert":            bool(alert),
        "per_feature":      per_feature,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    (out_dir / f"drift_report_{date}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    if alert:
        (out_dir / "drift_alert.json").write_text(
            json.dumps({"alert": True, "shifted_fraction": shifted_frac,
                        "n_shifted": shifted, "date": date}, indent=2),
            encoding="utf-8",
        )
        log.warning("DRIFT ALERT: %.1f%% of features shifted (threshold 5%%)", shifted_frac * 100)
    else:
        log.info("no drift: %.1f%% of features below p=%.2f", shifted_frac * 100, p_threshold)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default=str(ROOT / "predictions.db"))
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    run(Path(args.db), days=args.days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
