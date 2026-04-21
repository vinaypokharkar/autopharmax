"""Joint multitask training loop with MLflow logging.

Key invariants:
  - Joint loss: L = L_single + synergy_lambda * L_synergy
  - Synergy drug-order augmentation: swap (a, b) with p=0.5 each batch
  - Early stopping monitors val_r2 on the LEAVE-DRUG-OUT split, NEVER on
    the standard split. Overfitting to standard-split rows is what we're
    trying to avoid.
  - AdamW + CosineAnnealingLR, gradient-clipped at grad_clip_norm.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.multitask import MultiTaskWrapper

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    early_stopping_patience: int
    grad_clip_norm: float
    synergy_lambda: float


@torch.no_grad()
def eval_single(model, loader: DataLoader, device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    for c, d, y in loader:
        c = c.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        p = model.single_model(c, d).cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    if not ys:
        return {"r2": float("nan"), "pearson": float("nan"), "rmse": float("nan"), "n": 0}
    y = np.concatenate(ys); p = np.concatenate(ps)
    try:
        r2 = float(r2_score(y, p))
    except Exception:
        r2 = float("nan")
    try:
        pr = float(pearsonr(y, p)[0])
    except Exception:
        pr = float("nan")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {"r2": r2, "pearson": pr, "rmse": rmse, "n": int(len(y))}


def train(
    model: MultiTaskWrapper,
    single_train_loader: DataLoader,
    single_val_std_loader: DataLoader,
    single_val_ldo_loader: DataLoader,
    synergy_train_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: Path,
    run_name: str = "autopharmax_joint",
) -> dict[str, float]:

    model.to(device)
    optim = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(cfg.max_epochs, 1))

    best_ldo_r2 = -float("inf")
    patience = cfg.early_stopping_patience
    bad_epochs = 0
    best_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "grad_clip_norm": cfg.grad_clip_norm,
            "synergy_lambda": cfg.synergy_lambda,
            "cell_dim": model.cell_enc.net[0].in_features,
            "drug_dim": model.drug_enc.net[0].in_features,
        })

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = 0.0
            total_ls = 0.0
            total_lsy = 0.0
            n_steps = 0

            synergy_iter = iter(synergy_train_loader) if synergy_train_loader else None

            for cell, drug, y in single_train_loader:
                cell = cell.to(device, non_blocking=True)
                drug = drug.to(device, non_blocking=True)
                y    = y.to(device, non_blocking=True)

                pred_single = model.single_model(cell, drug)
                loss_single = F.mse_loss(pred_single, y)

                loss_synergy = torch.tensor(0.0, device=device)
                if synergy_iter is not None:
                    try:
                        sc, sa, sb, sy = next(synergy_iter)
                    except StopIteration:
                        synergy_iter = iter(synergy_train_loader)
                        sc, sa, sb, sy = next(synergy_iter)
                    if random.random() > 0.5:
                        sa, sb = sb, sa
                    sc = sc.to(device, non_blocking=True)
                    sa = sa.to(device, non_blocking=True)
                    sb = sb.to(device, non_blocking=True)
                    sy = sy.to(device, non_blocking=True)
                    pred_syn = model.synergy_model(sc, sa, sb)
                    loss_synergy = F.mse_loss(pred_syn, sy)

                loss = loss_single + cfg.synergy_lambda * loss_synergy

                # Defense-in-depth: skip a batch if the loss is not finite.
                # NaN/Inf would otherwise propagate into the shared encoders
                # via backward() and poison every subsequent forward pass.
                if not torch.isfinite(loss):
                    log.warning("non-finite loss at epoch=%d step=%d (single=%s syn=%s) - skipped",
                                epoch, n_steps, float(loss_single), float(loss_synergy))
                    optim.zero_grad(set_to_none=True)
                    continue

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()

                total_loss += float(loss.detach().cpu())
                total_ls   += float(loss_single.detach().cpu())
                total_lsy  += float(loss_synergy.detach().cpu())
                n_steps    += 1

            sched.step()

            std_metrics = eval_single(model, single_val_std_loader, device)
            ldo_metrics = eval_single(model, single_val_ldo_loader, device)

            log.info(
                "epoch %3d | loss=%.4f (single=%.4f syn=%.4f) | val R^2 std=%.3f ldo=%.3f | n_std=%d n_ldo=%d",
                epoch, total_loss / max(n_steps, 1), total_ls / max(n_steps, 1),
                total_lsy / max(n_steps, 1),
                std_metrics["r2"], ldo_metrics["r2"], std_metrics["n"], ldo_metrics["n"],
            )
            mlflow.log_metrics({
                "train_loss_total": total_loss / max(n_steps, 1),
                "train_loss_single": total_ls / max(n_steps, 1),
                "train_loss_synergy": total_lsy / max(n_steps, 1),
                "val_r2_standard": std_metrics["r2"],
                "val_pearson_standard": std_metrics["pearson"],
                "val_rmse_standard": std_metrics["rmse"],
                "val_r2_ldo": ldo_metrics["r2"],
                "val_pearson_ldo": ldo_metrics["pearson"],
                "val_rmse_ldo": ldo_metrics["rmse"],
                "lr": optim.param_groups[0]["lr"],
            }, step=epoch)

            # Early stopping on LDO R^2 only.
            if ldo_metrics["r2"] > best_ldo_r2:
                best_ldo_r2 = ldo_metrics["r2"]
                bad_epochs = 0
                best_metrics = {
                    "best_epoch": float(epoch),
                    "best_val_r2_ldo": float(ldo_metrics["r2"]),
                    "best_val_r2_standard": float(std_metrics["r2"]),
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_r2_ldo": ldo_metrics["r2"],
                    "val_r2_standard": std_metrics["r2"],
                }, checkpoint_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.info("early stopping at epoch %d (no LDO R^2 improvement for %d epochs)", epoch, patience)
                    break

        mlflow.log_metrics(best_metrics)
        log.info("best val_r2_ldo=%.4f at epoch %d", best_ldo_r2, int(best_metrics.get("best_epoch", -1)))
    return best_metrics
