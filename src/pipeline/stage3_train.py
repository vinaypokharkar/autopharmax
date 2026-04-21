"""Stage 3 - Joint multitask training.

Outputs:
  data/processed/best_checkpoint.pt
  MLflow run with all per-epoch metrics logged

Flags:
  --debug       subset both datasets to 2000 rows, max_epochs=3
  --cpu         force CPU (by default we assert CUDA available)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402

from src.models.multitask import MultiTaskWrapper  # noqa: E402
from src.training.datasets import (  # noqa: E402
    load_feature_caches, load_single_split, load_synergy_split,
)
from src.training.trainer import TrainConfig, train  # noqa: E402

log = logging.getLogger(__name__)
PROC = ROOT / "data" / "processed"


def _set_mlflow_tracking_uri():
    uri = f"file:///{(ROOT / 'mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("autopharmax")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    tp = params["training"]
    cfg = TrainConfig(
        batch_size=tp["batch_size"],
        max_epochs=tp["max_epochs"] if not args.debug else 3,
        learning_rate=tp["learning_rate"],
        weight_decay=tp["weight_decay"],
        dropout=tp["dropout"],
        early_stopping_patience=tp["early_stopping_patience"],
        grad_clip_norm=tp["grad_clip_norm"],
        synergy_lambda=tp["synergy_lambda"],
    )

    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            log.error("CUDA not available. Use --cpu to force CPU training.")
            return 2
        device = torch.device("cuda")
    log.info("device: %s", device)

    cell_feats, drug_feats = load_feature_caches()
    cell_dim = next(iter(cell_feats.values())).shape[0]
    drug_dim = next(iter(drug_feats.values())).shape[0]
    log.info("cell_dim=%d drug_dim=%d cells=%d drugs=%d",
             cell_dim, drug_dim, len(cell_feats), len(drug_feats))

    # Build datasets
    log.info("load splits")
    ds_train_std = load_single_split("standard", "train", cell_feats, drug_feats)
    ds_val_std   = load_single_split("standard", "val",   cell_feats, drug_feats)
    ds_val_ldo   = load_single_split("ldo",      "val",   cell_feats, drug_feats)
    syn_train    = load_synergy_split("train", cell_feats, drug_feats)

    if args.debug:
        log.info("--debug: subsetting datasets")
        ds_train_std = Subset(ds_train_std, range(min(2000, len(ds_train_std))))
        ds_val_std   = Subset(ds_val_std,   range(min(500,  len(ds_val_std))))
        ds_val_ldo   = Subset(ds_val_ldo,   range(min(500,  len(ds_val_ldo))))
        if syn_train is not None:
            syn_train = Subset(syn_train, range(min(1000, len(syn_train))))

    def _loader(ds, shuffle, bs=None):
        if ds is None:
            return None
        return DataLoader(
            ds, batch_size=(bs or cfg.batch_size), shuffle=shuffle,
            num_workers=0, pin_memory=(device.type == "cuda"), drop_last=shuffle,
        )

    dl_train_std = _loader(ds_train_std, shuffle=True)
    dl_val_std   = _loader(ds_val_std,   shuffle=False)
    dl_val_ldo   = _loader(ds_val_ldo,   shuffle=False)
    dl_syn_train = _loader(syn_train,    shuffle=True)
    log.info("loader sizes: single_train=%d val_std=%d val_ldo=%d syn_train=%s",
             len(ds_train_std), len(ds_val_std), len(ds_val_ldo),
             len(syn_train) if syn_train is not None else "None")

    # Model
    model = MultiTaskWrapper(cell_input_dim=cell_dim, drug_input_dim=drug_dim, params=params)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %d", n_params)

    # MLflow
    _set_mlflow_tracking_uri()

    ckpt = PROC / ("best_checkpoint_debug.pt" if args.debug else "best_checkpoint.pt")
    best = train(
        model=model,
        single_train_loader=dl_train_std,
        single_val_std_loader=dl_val_std,
        single_val_ldo_loader=dl_val_ldo,
        synergy_train_loader=dl_syn_train,
        cfg=cfg,
        device=device,
        checkpoint_path=ckpt,
        run_name=("autopharmax_debug" if args.debug else "autopharmax_joint"),
    )
    log.info("Stage 3 complete. checkpoint=%s best=%s", ckpt, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
