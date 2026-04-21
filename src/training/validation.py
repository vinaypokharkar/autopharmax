"""Four-split evaluation for AutoPharmaX.

Loads best_checkpoint.pt and reports per-split metrics on:
  - standard (random test rows)
  - ldo  (leave-drug-out test, the HEADLINE metric)
  - lclo (leave-cell-line-out test)
  - lpo  (synergy leave-pair-out test, if synergy data is available)

Produces scatter PNGs + error distribution PNG + a metrics dict.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models.multitask import MultiTaskWrapper
from src.training.datasets import (
    load_feature_caches, load_single_split, load_synergy_split,
)

log = logging.getLogger(__name__)


def _regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"r2": float("nan"), "pearson": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "n": 0}
    r2 = float(r2_score(y, p))
    try:
        pr = float(pearsonr(y, p)[0])
    except Exception:
        pr = float("nan")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae  = float(np.mean(np.abs(y - p)))
    return {"r2": r2, "pearson": pr, "rmse": rmse, "mae": mae, "n": int(len(y))}


@torch.no_grad()
def _predict_single(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for c, d, y in loader:
        c = c.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        ps.append(model.single_model(c, d).cpu().numpy())
        ys.append(y.numpy())
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def _predict_synergy(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for c, a, b, y in loader:
        c = c.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        ps.append(model.synergy_model(c, a, b).cpu().numpy())
        ys.append(y.numpy())
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def _scatter(y: np.ndarray, p: np.ndarray, title: str, path: Path) -> None:
    if len(y) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=3, alpha=0.25)
    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _error_hist(splits: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (y, p) in splits.items():
        if len(y) == 0:
            continue
        errs = p - y
        ax.hist(errs, bins=60, histtype="step", label=f"{name} (n={len(y)})", linewidth=1.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("prediction error (pred - actual)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_all(
    checkpoint_path: Path,
    out_dir: Path,
    device: torch.device,
    cell_feats: dict | None = None,
    drug_feats: dict | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cell_feats is None or drug_feats is None:
        cell_feats, drug_feats = load_feature_caches()
    cell_dim = next(iter(cell_feats.values())).shape[0]
    drug_dim = next(iter(drug_feats.values())).shape[0]

    model = MultiTaskWrapper(cell_input_dim=cell_dim, drug_input_dim=drug_dim, params=params)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    metrics: dict[str, Any] = {"checkpoint": str(checkpoint_path),
                               "best_epoch": int(state.get("epoch", -1))}

    # ---- Single-drug splits -------------------------------------------------
    scatters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("standard", "ldo", "lclo"):
        ds = load_single_split(name, "test", cell_feats, drug_feats)
        loader = DataLoader(ds, batch_size=1024, shuffle=False,
                            pin_memory=(device.type == "cuda"))
        y, p = _predict_single(model, loader, device)
        m = _regression_metrics(y, p)
        metrics[name] = m
        scatters[name] = (y, p)
        _scatter(y, p, f"{name.upper()} test (R^2 = {m['r2']:.3f}, n={m['n']})",
                 out_dir / f"predicted_vs_actual_{name}.png")
        log.info("%-8s test: R^2=%.4f pearson=%.4f rmse=%.4f mae=%.4f n=%d",
                 name, m["r2"], m["pearson"], m["rmse"], m["mae"], m["n"])

    # ---- Synergy LPO --------------------------------------------------------
    ds_syn = load_synergy_split("test", cell_feats, drug_feats)
    if ds_syn is None or len(ds_syn) == 0:
        log.warning("synergy LPO test set empty - skipping")
        metrics["lpo"] = {"r2": float("nan"), "pearson": float("nan"),
                          "rmse": float("nan"), "mae": float("nan"),
                          "auroc": float("nan"), "n": 0}
    else:
        loader = DataLoader(ds_syn, batch_size=1024, shuffle=False,
                            pin_memory=(device.type == "cuda"))
        y, p = _predict_synergy(model, loader, device)
        m = _regression_metrics(y, p)
        # AUROC treating Loewe > 10 as synergistic (class balance permitting)
        y_bin = (y > 10).astype(int)
        if y_bin.sum() >= 5 and (len(y_bin) - y_bin.sum()) >= 5:
            try:
                m["auroc"] = float(roc_auc_score(y_bin, p))
            except Exception as e:
                log.warning("AUROC failed: %s", e)
                m["auroc"] = float("nan")
        else:
            m["auroc"] = float("nan")
            log.info("synergy LPO class imbalance too extreme for AUROC (n_pos=%d)",
                     int(y_bin.sum()))
        metrics["lpo"] = m
        scatters["lpo"] = (y, p)
        _scatter(y, p, f"Synergy LPO test (R^2 = {m['r2']:.3f}, n={m['n']})",
                 out_dir / "predicted_vs_actual_lpo.png")
        log.info("lpo     test: R^2=%.4f pearson=%.4f rmse=%.4f mae=%.4f auroc=%.4f n=%d",
                 m["r2"], m["pearson"], m["rmse"], m["mae"], m.get("auroc", float("nan")), m["n"])

    # ---- Error distribution ------------------------------------------------
    _error_hist(scatters, out_dir / "error_distribution.png")

    return metrics


def print_comparison_table(metrics: dict[str, Any]) -> None:
    """Print the 4-row results table demanded by the spec."""
    rows = [
        ("Standard test",              metrics.get("standard", {})),
        ("Leave-drug-out (headline)",  metrics.get("ldo", {})),
        ("Leave-cell-line-out",        metrics.get("lclo", {})),
        ("Synergy leave-pair-out",     metrics.get("lpo", {})),
    ]
    print()
    print("+-----------------------------+-------+----------+-----------+-----------+")
    print("| Split                       |  R^2  | Pearson  |    RMSE   |    MAE    |")
    print("+-----------------------------+-------+----------+-----------+-----------+")
    for name, m in rows:
        print(f"| {name:<27s} | {m.get('r2', float('nan')):.3f} |  {m.get('pearson', float('nan')):.3f}   |  {m.get('rmse', float('nan')):.4f}  |  {m.get('mae', float('nan')):.4f}  |")
    print("+-----------------------------+-------+----------+-----------+-----------+")
    print()
