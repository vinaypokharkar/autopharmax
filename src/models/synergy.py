"""Drug combination synergy model.

Consumes one cell context + two drug contexts (same DrugEncoder instance
used for both, giving order-agnostic weight sharing after augmentation).
Output is predicted Loewe synergy score.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import CellLineEncoder, DrugEncoder, RegressionHead


class AutoPharmaXSynergy(nn.Module):
    def __init__(self, cell_enc: CellLineEncoder, drug_enc: DrugEncoder, head_dims: list[int]):
        super().__init__()
        self.cell_enc = cell_enc
        self.drug_enc = drug_enc  # SAME instance as single-drug model
        fusion_dim = cell_enc.output_dim + 2 * drug_enc.output_dim
        self.head = RegressionHead(fusion_dim, head_dims)

    def forward(self, cell_feats: torch.Tensor, drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        c = self.cell_enc(cell_feats)
        a = self.drug_enc(drug_a)
        b = self.drug_enc(drug_b)
        return self.head(torch.cat([c, a, b], dim=1))
