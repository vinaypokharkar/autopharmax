"""Singleton inference service: loads the best checkpoint once, holds
feature caches in memory, and answers prediction queries.

Also handles on-the-fly drug featurisation: if a drug isn't in the
precomputed cache, we fetch its SMILES from PubChem, run RDKit, apply
the training-time scaler, and cache the result in-process.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"

log = logging.getLogger(__name__)


class InferenceService:
    def __init__(self) -> None:
        self.params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature caches.
        with open(PROC / "cell_features.pkl", "rb") as f:
            self.cell_feats: dict[int, np.ndarray] = pickle.load(f)
        with open(PROC / "drug_features.pkl", "rb") as f:
            self.drug_feats: dict[str, np.ndarray] = pickle.load(f)
        self.cell_dim = next(iter(self.cell_feats.values())).shape[0]
        self.drug_dim = next(iter(self.drug_feats.values())).shape[0]

        # Drug descriptor scaler + medians (for on-the-fly featurisation)
        with open(PROC / "drug_desc_scaler.pkl", "rb") as f:
            desc_payload = pickle.load(f)
        self.desc_scaler = desc_payload["scaler"]
        self.desc_medians = desc_payload["medians"]
        self.desc_names = desc_payload["desc_names"]

        # Metadata
        self.meta = json.loads((PROC / "feature_metadata.json").read_text(encoding="utf-8"))
        self.morgan_bits = int(self.meta.get("morgan_bits", 2048))

        # Cell-line tissue info (for /cell-lines endpoint).
        self.cell_line_info = self._load_cell_line_info()

        # Known-pair lookup (informational only; never used for prediction).
        single = pd.read_parquet(PROC / "merged_single.parquet")
        self.known_pairs: dict[tuple[int, str], float] = {
            (int(r["COSMIC_ID"]), str(r["DRUG_NAME"])): float(r["LN_IC50"])
            for _, r in single.iterrows()
        }

        # SHAP importance caches (loaded once; global attribution summary).
        self.cell_shap = json.loads((PROC / "shap_cell_importance.json").read_text(encoding="utf-8"))
        self.drug_shap = json.loads((PROC / "shap_drug_importance.json").read_text(encoding="utf-8"))

        # Build + load model.
        from src.models.multitask import MultiTaskWrapper
        self.model = MultiTaskWrapper(
            cell_input_dim=self.cell_dim, drug_input_dim=self.drug_dim, params=self.params,
        )
        ckpt = torch.load(PROC / "best_checkpoint.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.best_epoch = int(ckpt.get("epoch", -1))

        # Metrics + MLflow Production version (best-effort).
        self.metrics: dict = {}
        metrics_path = ROOT / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.model_version = self._read_production_version()

    # ---- Cell-line catalogue --------------------------------------------

    def _load_cell_line_info(self) -> dict[int, dict]:
        info: dict[int, dict] = {}
        dm_path = RAW / "depmap_model.csv"
        if not dm_path.exists():
            return info
        df = pd.read_csv(dm_path, usecols=["ModelID", "StrippedCellLineName", "CellLineName",
                                           "OncotreeLineage", "COSMICID"])
        df = df.dropna(subset=["COSMICID"]).copy()
        df["COSMIC_ID"] = pd.to_numeric(df["COSMICID"], errors="coerce").round().astype("Int64")
        df = df.dropna(subset=["COSMIC_ID"]).copy()
        df["COSMIC_ID"] = df["COSMIC_ID"].astype(int)
        for _, r in df.iterrows():
            info[int(r["COSMIC_ID"])] = {
                "cell_line_name": str(r["StrippedCellLineName"])
                                  if pd.notna(r.get("StrippedCellLineName"))
                                  else str(r.get("CellLineName", "") or ""),
                "tissue":         str(r.get("OncotreeLineage", "") or "") or None,
            }
        return info

    # ---- MLflow Production version -------------------------------------

    def _read_production_version(self) -> Optional[str]:
        try:
            import mlflow
            mlflow.set_tracking_uri(f"file:///{(ROOT / 'mlruns').resolve().as_posix()}")
            client = mlflow.MlflowClient()
            versions = client.search_model_versions("name='AutoPharmaX_SingleDrug'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if prod:
                prod.sort(key=lambda v: int(v.version), reverse=True)
                return f"v{prod[0].version}"
        except Exception as e:
            log.warning("MLflow version lookup failed: %s", e)
        return None

    # ---- Drug featurisation on the fly ----------------------------------

    def get_drug_features(self, drug_name: str) -> Optional[np.ndarray]:
        """Return the cached or freshly-computed drug feature vector.
        Returns None if PubChem + RDKit can't produce features."""
        if drug_name in self.drug_feats:
            return self.drug_feats[drug_name]

        smiles = self._fetch_smiles(drug_name)
        if not smiles:
            return None
        vec = self._featurise_smiles(smiles)
        if vec is None:
            return None
        # In-memory cache only; we don't mutate drug_features.pkl at runtime.
        self.drug_feats[drug_name] = vec
        return vec

    def _fetch_smiles(self, name: str) -> Optional[str]:
        # Check data/raw/drug_smiles.csv first (fast)
        smi_path = RAW / "drug_smiles.csv"
        if smi_path.exists():
            df = pd.read_csv(smi_path)
            hit = df[df["drug_name"].astype(str).str.lower() == name.lower()]
            if len(hit):
                return str(hit.iloc[0]["smiles"])
        # Else hit PubChem
        try:
            import requests
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{requests.utils.quote(name)}/property/SMILES/JSON"
            )
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return None
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if not props:
                return None
            p = props[0]
            return p.get("SMILES") or p.get("IsomericSMILES") or p.get("CanonicalSMILES")
        except Exception as e:
            log.warning("PubChem lookup failed for %s: %s", name, e)
            return None

    def _featurise_smiles(self, smiles: str) -> Optional[np.ndarray]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit.Chem import rdFingerprintGenerator
        except Exception as e:
            log.error("rdkit import failed: %s", e)
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=int(self.params["data"]["morgan_radius"]),
                fpSize=int(self.morgan_bits),
            )
            morgan = np.array(mgen.GetFingerprint(mol), dtype=np.float32)

            # Compute the SAME descriptor set by NAME (not position), so we
            # honour the training-time drop list.
            desc_fn_by_name = {d[0]: d[1] for d in Descriptors.descList}
            vals = []
            for n in self.desc_names:
                fn = desc_fn_by_name.get(n)
                if fn is None:
                    vals.append(np.nan); continue
                try:
                    vals.append(float(fn(mol)))
                except Exception:
                    vals.append(np.nan)
            arr = np.asarray(vals, dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            # impute with training medians
            arr = np.where(np.isnan(arr), self.desc_medians, arr)
            # scale
            arr = self.desc_scaler.transform(arr.reshape(1, -1)).astype(np.float32).ravel()
        return np.concatenate([morgan, arr]).astype(np.float32)

    # ---- Inference ------------------------------------------------------

    @torch.no_grad()
    def predict_single(self, cosmic_id: int, drug_name: str) -> dict:
        t0 = time.time()
        cell = self.cell_feats.get(int(cosmic_id))
        if cell is None:
            raise KeyError(f"cosmic_id {cosmic_id} not in cell feature cache")
        drug = self.get_drug_features(drug_name)
        if drug is None:
            raise ValueError(f"could not featurise drug '{drug_name}' (no SMILES / RDKit parse)")
        c = torch.from_numpy(cell).unsqueeze(0).to(self.device)
        d = torch.from_numpy(drug).unsqueeze(0).to(self.device)
        pred = float(self.model.single_model(c, d).cpu().numpy()[0])
        known = self.known_pairs.get((int(cosmic_id), drug_name))
        ic50_uM = float(np.exp(pred))
        # Dataset median LN_IC50 is ~3.22 (from stage2 summary) - we cache
        # nothing here; use predicted <= 3.0 as "sensitive".
        sensitivity = "sensitive" if pred <= 3.0 else "insensitive"
        return {
            "predicted_ln_ic50": pred,
            "predicted_ic50_uM": ic50_uM,
            "sensitivity_class": sensitivity,
            "is_novel_prediction": known is None,
            "actual_ln_ic50": known,
            "absolute_error": abs(pred - known) if known is not None else None,
            "top_cell_features": self.cell_shap[:10],
            "top_drug_features": self.drug_shap[:10],
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    @torch.no_grad()
    def predict_combination(self, cosmic_id: int, drug_a: str, drug_b: str) -> dict:
        t0 = time.time()
        cell = self.cell_feats.get(int(cosmic_id))
        if cell is None:
            raise KeyError(f"cosmic_id {cosmic_id} not in cell feature cache")
        a = self.get_drug_features(drug_a)
        b = self.get_drug_features(drug_b)
        if a is None:
            raise ValueError(f"could not featurise drug_a '{drug_a}'")
        if b is None:
            raise ValueError(f"could not featurise drug_b '{drug_b}'")
        c = torch.from_numpy(cell).unsqueeze(0).to(self.device)
        ta = torch.from_numpy(a).unsqueeze(0).to(self.device)
        tb = torch.from_numpy(b).unsqueeze(0).to(self.device)
        syn = float(self.model.synergy_model(c, ta, tb).cpu().numpy()[0])
        ic_a = float(self.model.single_model(c, ta).cpu().numpy()[0])
        ic_b = float(self.model.single_model(c, tb).cpu().numpy()[0])
        cls = ("synergistic" if syn > 10 else
               "antagonistic" if syn < -10 else "additive")
        return {
            "synergy_score": syn,
            "synergy_class": cls,
            "drug_a_ln_ic50": ic_a,
            "drug_b_ln_ic50": ic_b,
            "is_novel_combination": True,
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    @torch.no_grad()
    def combination_scan(self, cosmic_id: int, drug_a: str, top_k: int) -> dict:
        t0 = time.time()
        cell = self.cell_feats.get(int(cosmic_id))
        if cell is None:
            raise KeyError(f"cosmic_id {cosmic_id} not in cell feature cache")
        a = self.get_drug_features(drug_a)
        if a is None:
            raise ValueError(f"could not featurise drug_a '{drug_a}'")

        # Stack all cached drugs as candidate drug_b.
        names = [n for n in self.drug_feats.keys() if n != drug_a]
        b_vecs = np.stack([self.drug_feats[n] for n in names])
        n = len(names)
        c_rep = np.tile(cell, (n, 1))
        a_rep = np.tile(a,    (n, 1))

        c_t = torch.from_numpy(c_rep).to(self.device)
        a_t = torch.from_numpy(a_rep).to(self.device)
        b_t = torch.from_numpy(b_vecs).to(self.device)

        # Forward in chunks to bound memory even on big drug catalogues.
        chunk = 1024
        scores = np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            scores[s:e] = self.model.synergy_model(c_t[s:e], a_t[s:e], b_t[s:e]).cpu().numpy()

        order = np.argsort(-scores)[:top_k]
        results = []
        for i in order:
            s = float(scores[i])
            cls = ("synergistic" if s > 10 else
                   "antagonistic" if s < -10 else "additive")
            results.append({"drug_b": names[i], "synergy_score": s, "synergy_class": cls})
        return {
            "cosmic_id": int(cosmic_id),
            "drug_a": drug_a,
            "top_k": top_k,
            "results": results,
            "latency_ms": (time.time() - t0) * 1000.0,
        }
