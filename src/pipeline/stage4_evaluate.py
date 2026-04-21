"""Stage 4 - Evaluation across all four splits + quality gate.

Loads data/processed/best_checkpoint.pt, runs standard/LDO/LCLO/synergy-LPO
test sets, writes metrics.json, and hard-fails (exit 1) if LDO R^2 falls
below the quality_gate.min_ldo_r2 threshold from params.yaml.

Also logs every number + artifact to the MLflow run so Stage 5 can
promote the model version.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.validation import evaluate_all, print_comparison_table  # noqa: E402

log = logging.getLogger(__name__)
PROC = ROOT / "data" / "processed"


def _set_mlflow_tracking_uri():
    uri = f"file:///{(ROOT / 'mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("autopharmax")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROC / "best_checkpoint.pt"))
    ap.add_argument("--out-dir",     default=str(PROC / "eval"))
    ap.add_argument("--metrics",     default=str(ROOT / "metrics.json"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    min_ldo_r2 = float(params["quality_gate"]["min_ldo_r2"])

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    log.info("device: %s", device)

    checkpoint = Path(args.checkpoint)
    out_dir    = Path(args.out_dir)
    if not checkpoint.exists():
        log.error("checkpoint not found: %s (run Stage 3 first)", checkpoint)
        return 2

    metrics = evaluate_all(checkpoint_path=checkpoint, out_dir=out_dir,
                            device=device, params=params)

    print_comparison_table(metrics)

    # Log to MLflow as a new run tagged as evaluation-only
    _set_mlflow_tracking_uri()
    with mlflow.start_run(run_name="autopharmax_eval"):
        flat = {}
        for split, m in metrics.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        flat[f"{split}_{k}"] = v
        mlflow.log_metrics(flat)
        for png in sorted(out_dir.glob("*.png")):
            mlflow.log_artifact(str(png))

    # Write metrics.json (DVC metrics target)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("metrics written -> %s", args.metrics)

    # Quality gate
    ldo_r2 = metrics.get("ldo", {}).get("r2", float("nan"))
    if not (ldo_r2 == ldo_r2):   # NaN check
        log.error("GATE FAILED: LDO R^2 is NaN")
        return 1
    if ldo_r2 < min_ldo_r2:
        log.error("GATE FAILED: LDO R^2=%.4f < threshold %.2f", ldo_r2, min_ldo_r2)
        return 1
    log.info("GATE PASSED: LDO R^2=%.4f >= threshold %.2f", ldo_r2, min_ldo_r2)
    log.info("Stage 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
