"""SHAP feature attribution for the single-drug IC50 model.

We use shap.GradientExplainer rather than KernelExplainer - it's ~100x
faster on 3630-dim inputs and our model is fully differentiable (BN in
eval mode is just an affine transform, which SHAP handles cleanly).
DeepExplainer chokes on our concat-shaped forward pass, so it's out.

Outputs:
  data/processed/shap_cell_importance.json     top-20 genes by mean |SHAP|
  data/processed/shap_drug_importance.json     top-20 descriptors by mean |SHAP|
  data/processed/eval/shap_top_genes.png       bar plot
  data/processed/eval/shap_top_descriptors.png bar plot

The Morgan fingerprint block (2048 bits) is excluded from the "top drug
features" view - individual bits aren't interpretable by name.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.multitask import MultiTaskWrapper  # noqa: E402
from src.training.datasets import (  # noqa: E402
    load_feature_caches, load_single_split,
)

log = logging.getLogger(__name__)
PROC = ROOT / "data" / "processed"
EVAL_DIR = PROC / "eval"


class _ConcatSingleDrug(torch.nn.Module):
    """Wrap AutoPharmaXSingleDrug to accept a single concatenated tensor.

    SHAP GradientExplainer expects one tensor input per branch; wrapping
    lets us treat the whole [cell, drug] vector as one input for
    attribution.
    """
    def __init__(self, single_model, cell_dim: int):
        super().__init__()
        self.single_model = single_model
        self.cell_dim = cell_dim

    def forward(self, x):
        cell = x[:, :self.cell_dim]
        drug = x[:, self.cell_dim:]
        return self.single_model(cell, drug).unsqueeze(-1)


def _stack_batches(ds, indices):
    """Materialise a (len(indices), cell+drug) tensor from a dataset."""
    rows = [ds[i] for i in indices]
    c = torch.stack([r[0] for r in rows])
    d = torch.stack([r[1] for r in rows])
    return torch.cat([c, d], dim=1)


def compute_shap(
    n_background: int = 50,
    n_explain: int = 100,
    split: str = "ldo",
    part: str = "test",
    seed: int = 0,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    cell_feats, drug_feats = load_feature_caches()
    cell_dim = next(iter(cell_feats.values())).shape[0]
    drug_dim = next(iter(drug_feats.values())).shape[0]

    # Load trained model + wrap.
    model = MultiTaskWrapper(cell_input_dim=cell_dim, drug_input_dim=drug_dim, params=params)
    state = torch.load(PROC / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    concat_model = _ConcatSingleDrug(model.single_model, cell_dim).to(device).eval()

    # Background from TRAIN (standard split), explainees from split/part.
    ds_train = load_single_split("standard", "train", cell_feats, drug_feats)
    ds_test  = load_single_split(split,      part,    cell_feats, drug_feats)
    rng = np.random.default_rng(seed)
    bg_idx   = rng.choice(len(ds_train), size=min(n_background, len(ds_train)), replace=False)
    test_idx = rng.choice(len(ds_test),  size=min(n_explain,    len(ds_test)),  replace=False)

    bg   = _stack_batches(ds_train, bg_idx).to(device)
    expl = _stack_batches(ds_test,  test_idx).to(device)
    log.info("bg shape %s, explain shape %s", tuple(bg.shape), tuple(expl.shape))

    explainer = shap.GradientExplainer(concat_model, bg)
    raw = explainer.shap_values(expl)
    # shap may return a list-of-one-array or a single array depending on output shape
    if isinstance(raw, list):
        shap_vals = raw[0]
    else:
        shap_vals = raw
    shap_vals = np.asarray(shap_vals)
    # Collapse a trailing singleton dim (model output is (N, 1) after unsqueeze)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[..., 0]
    log.info("shap values shape: %s", shap_vals.shape)

    # Split into cell and drug halves.
    meta = json.loads((PROC / "feature_metadata.json").read_text(encoding="utf-8"))
    cell_names       = meta["cell_features"]
    desc_names       = meta["drug_descriptors"]
    morgan_bits      = int(meta["morgan_bits"])

    cell_shap = np.abs(shap_vals[:, :cell_dim]).mean(axis=0)
    drug_shap = np.abs(shap_vals[:, cell_dim:]).mean(axis=0)
    drug_desc_shap = drug_shap[morgan_bits:]     # skip Morgan bits - not interpretable

    top_genes = sorted(zip(cell_names, cell_shap),      key=lambda x: -x[1])[:20]
    top_descs = sorted(zip(desc_names, drug_desc_shap), key=lambda x: -x[1])[:20]

    # Save JSON
    (PROC / "shap_cell_importance.json").write_text(
        json.dumps([{"feature": n, "importance": float(v)} for n, v in top_genes], indent=2),
        encoding="utf-8",
    )
    (PROC / "shap_drug_importance.json").write_text(
        json.dumps([{"feature": n, "importance": float(v)} for n, v in top_descs], indent=2),
        encoding="utf-8",
    )

    # Bar plots
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _bar([x[0] for x in top_genes], [x[1] for x in top_genes],
         f"Top 20 cell features (SHAP |mean|) - {split}-{part} n={n_explain}",
         EVAL_DIR / "shap_top_genes.png")
    _bar([x[0] for x in top_descs], [x[1] for x in top_descs],
         f"Top 20 drug descriptors (SHAP |mean|) - {split}-{part} n={n_explain}",
         EVAL_DIR / "shap_top_descriptors.png")

    return {
        "n_background": int(bg.shape[0]),
        "n_explained":  int(expl.shape[0]),
        "top_cell_features":     [{"feature": n, "importance": float(v)} for n, v in top_genes],
        "top_drug_descriptors":  [{"feature": n, "importance": float(v)} for n, v in top_descs],
    }


def _bar(names, values, title: str, path: Path) -> None:
    order = list(range(len(names)))[::-1]   # largest at top
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order], [values[i] for i in order])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-background", type=int, default=50)
    ap.add_argument("--n-explain",    type=int, default=100)
    ap.add_argument("--split", default="ldo", choices=["standard", "ldo", "lclo"])
    ap.add_argument("--part",  default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    out = compute_shap(n_background=args.n_background, n_explain=args.n_explain,
                       split=args.split, part=args.part)
    print(json.dumps(out, indent=2)[:2000])

    # Log summary bar plots to MLflow
    import mlflow
    mlflow.set_tracking_uri(f"file:///{(ROOT / 'mlruns').resolve().as_posix()}")
    mlflow.set_experiment("autopharmax")
    with mlflow.start_run(run_name="autopharmax_shap"):
        for png in sorted(EVAL_DIR.glob("shap_top_*.png")):
            mlflow.log_artifact(str(png))
        for fname in ("shap_cell_importance.json", "shap_drug_importance.json"):
            mlflow.log_artifact(str(PROC / fname))

    return 0


if __name__ == "__main__":
    sys.exit(main())
