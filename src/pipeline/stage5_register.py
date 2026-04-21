"""Stage 5 - MLflow model registry.

Logs both single-drug and synergy models under a single MLflow run,
attaches the scalers + feature_metadata.json as artifacts so downstream
environments can reconstruct inputs, and registers the two models as:
  - AutoPharmaX_SingleDrug
  - AutoPharmaX_Synergy

Each new version is first transitioned to Staging. If metrics.json shows
LDO R^2 >= quality_gate.min_ldo_r2, the version is promoted to Production
and any previous Production version is archived.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.multitask import MultiTaskWrapper  # noqa: E402

log = logging.getLogger(__name__)
PROC = ROOT / "data" / "processed"

SINGLE_NAME  = "AutoPharmaX_SingleDrug"
SYNERGY_NAME = "AutoPharmaX_Synergy"


def _set_mlflow_tracking_uri():
    uri = f"file:///{(ROOT / 'mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("autopharmax")


def _promote(client: mlflow.MlflowClient, model_name: str, run_id: str,
             to_production: bool) -> int:
    """Register a new version and transition it. Returns the version number."""
    model_uri = f"runs:/{run_id}/{model_name}"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    version = int(mv.version)
    # Staging first (unconditional)
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    if to_production:
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production",
            archive_existing_versions=True,
        )
        log.info("%s v%d -> Production", model_name, version)
    else:
        log.info("%s v%d -> Staging (quality gate failed; not promoted)", model_name, version)
    return version


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROC / "best_checkpoint.pt"))
    ap.add_argument("--metrics",    default=str(ROOT / "metrics.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    checkpoint = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    if not checkpoint.exists():
        log.error("checkpoint not found: %s (run Stage 3)", checkpoint)
        return 2
    if not metrics_path.exists():
        log.error("metrics not found: %s (run Stage 4)", metrics_path)
        return 2

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    min_ldo_r2 = float(params["quality_gate"]["min_ldo_r2"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ldo_r2 = float(metrics.get("ldo", {}).get("r2", float("nan")))
    to_production = ldo_r2 == ldo_r2 and ldo_r2 >= min_ldo_r2
    log.info("LDO R^2=%.4f, threshold=%.2f -> production=%s",
             ldo_r2, min_ldo_r2, to_production)

    # Rebuild model + load weights for logging.
    import pickle
    with open(PROC / "cell_features.pkl", "rb") as f:
        cell_feats = pickle.load(f)
    with open(PROC / "drug_features.pkl", "rb") as f:
        drug_feats = pickle.load(f)
    cell_dim = next(iter(cell_feats.values())).shape[0]
    drug_dim = next(iter(drug_feats.values())).shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskWrapper(cell_input_dim=cell_dim, drug_input_dim=drug_dim, params=params)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    _set_mlflow_tracking_uri()
    client = mlflow.MlflowClient()

    # Ensure registered-model names exist (idempotent).
    for name in (SINGLE_NAME, SYNERGY_NAME):
        try:
            client.create_registered_model(name)
            log.info("registered model created: %s", name)
        except mlflow.exceptions.RestException:
            pass
        except Exception as e:
            # MLflow local backend raises MlflowException for "already exists"
            if "already exists" not in str(e).lower():
                log.warning("create_registered_model(%s) non-fatal: %s", name, e)

    with mlflow.start_run(run_name="autopharmax_register") as run:
        run_id = run.info.run_id

        # Log artifacts
        for fname in ("cell_scaler.pkl", "drug_desc_scaler.pkl", "feature_metadata.json"):
            fp = PROC / fname
            if fp.exists():
                mlflow.log_artifact(str(fp))
        mlflow.log_artifact(str(metrics_path))

        # Log the two sub-models separately so they can be served independently.
        # We skip input_example because mlflow's PyTorch flavor doesn't accept
        # dict-valued examples for multi-arg forward() signatures.
        mlflow.pytorch.log_model(pytorch_model=model.single_model,
                                 artifact_path=SINGLE_NAME)
        mlflow.pytorch.log_model(pytorch_model=model.synergy_model,
                                 artifact_path=SYNERGY_NAME)

        # Log headline metrics to this registration run so it's readable
        # from the registry UI.
        flat = {}
        for split, m in metrics.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        flat[f"{split}_{k}"] = v
        mlflow.log_metrics(flat)
        mlflow.log_params({
            "cell_feature_dim": cell_dim,
            "drug_feature_dim": drug_dim,
            "min_ldo_r2_gate":  min_ldo_r2,
            "best_epoch":       int(state.get("epoch", -1)),
        })

        single_v = _promote(client, SINGLE_NAME,  run_id, to_production=to_production)
        syn_v    = _promote(client, SYNERGY_NAME, run_id, to_production=to_production)

        log.info("=" * 60)
        log.info("Stage 5 registration summary")
        log.info("  run_id: %s", run_id)
        log.info("  %s: v%d (%s)", SINGLE_NAME,  single_v,
                 "Production" if to_production else "Staging")
        log.info("  %s: v%d (%s)", SYNERGY_NAME, syn_v,
                 "Production" if to_production else "Staging")
        log.info("  LDO R^2=%.4f  Standard R^2=%.4f  Synergy LPO R^2=%.4f",
                 ldo_r2,
                 float(metrics.get("standard", {}).get("r2", float("nan"))),
                 float(metrics.get("lpo", {}).get("r2", float("nan"))))
        log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
