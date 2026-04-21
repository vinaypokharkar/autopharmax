"""Single-drug IC50 prediction model.

Takes a CellLineEncoder + DrugEncoder (possibly shared with the synergy
model) and an MLP regression head. Output is predicted LN(IC50).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import CellLineEncoder, DrugEncoder, RegressionHead


class AutoPharmaXSingleDrug(nn.Module):
    def __init__(self, cell_enc: CellLineEncoder, drug_enc: DrugEncoder, head_dims: list[int]):
        super().__init__()
        self.cell_enc = cell_enc
        self.drug_enc = drug_enc
        fusion_dim = cell_enc.output_dim + drug_enc.output_dim
        self.head = RegressionHead(fusion_dim, head_dims)

    def forward(self, cell_feats: torch.Tensor, drug_feats: torch.Tensor) -> torch.Tensor:
        c = self.cell_enc(cell_feats)
        d = self.drug_enc(drug_feats)
        return self.head(torch.cat([c, d], dim=1))
