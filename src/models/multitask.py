"""Multitask wrapper: owns one CellLineEncoder + one DrugEncoder and both heads.

The single-drug and synergy models share the SAME encoder instances (same
object id), not merely the same architecture. Gradients from both tasks
flow into the same encoder weights, which is the point of multitask
training here.
"""
from __future__ import annotations

import torch.nn as nn
import yaml
from pathlib import Path

from .encoders import CellLineEncoder, DrugEncoder
from .single_drug import AutoPharmaXSingleDrug
from .synergy import AutoPharmaXSynergy


def _load_params() -> dict:
    return yaml.safe_load((Path(__file__).resolve().parents[2] / "params.yaml").read_text(encoding="utf-8"))


class MultiTaskWrapper(nn.Module):
    def __init__(self, cell_input_dim: int, drug_input_dim: int, params: dict | None = None):
        super().__init__()
        p = params or _load_params()
        mp = p["model"]
        dropout = p["training"]["dropout"]

        self.cell_enc = CellLineEncoder(cell_input_dim, mp["cell_hidden"], dropout=dropout)
        self.drug_enc = DrugEncoder(drug_input_dim, mp["drug_hidden"], dropout=dropout)

        self.single_model = AutoPharmaXSingleDrug(self.cell_enc, self.drug_enc, mp["single_head"])
        self.synergy_model = AutoPharmaXSynergy(self.cell_enc, self.drug_enc, mp["synergy_head"])

        # Integrity assertion for the weight-sharing invariant.
        assert id(self.single_model.drug_enc) == id(self.synergy_model.drug_enc), \
            "DrugEncoder must be a single shared instance across single + synergy heads"
        assert id(self.single_model.cell_enc) == id(self.synergy_model.cell_enc), \
            "CellLineEncoder must be a single shared instance across single + synergy heads"
