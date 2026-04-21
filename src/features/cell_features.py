"""Cell-line feature vector construction: 978 L1000 + 200 mutation + 200 CNV = 1378 dims.

Output: {cosmic_id: np.ndarray(1378)} pickled to data/processed/cell_features.pkl
        + StandardScaler fit on TRAIN indices only -> data/processed/cell_scaler.pkl
        + ordered feature names -> data/processed/feature_metadata.json (cell_features field)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def _load_l1000_genes() -> list[str]:
    path = RAW / "l1000_genes.txt"
    genes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    log.info("L1000 gene list: %d symbols", len(genes))
    return genes


def _load_depmap_model() -> pd.DataFrame:
    """Load DepMap Model.csv with both DepMap_ID and COSMIC_ID columns."""
    df = pd.read_csv(RAW / "depmap_model.csv")
    # DepMap renames columns between releases. Normalise:
    rename = {c: c for c in df.columns}
    for candidate in ["ModelID", "DepMap_ID", "depmap_id"]:
        if candidate in df.columns:
            rename[candidate] = "DepMap_ID"
            break
    for candidate in ["COSMICID", "COSMIC_ID", "cosmic_id", "SangerModelID"]:
        if candidate in df.columns:
            rename[candidate] = "COSMIC_ID"
            break
    df = df.rename(columns=rename)
    if "DepMap_ID" not in df.columns or "COSMIC_ID" not in df.columns:
        raise AssertionError(f"depmap_model.csv missing DepMap_ID/COSMIC_ID (have: {list(df.columns)[:10]}...)")
    df = df.dropna(subset=["DepMap_ID", "COSMIC_ID"]).copy()
    # COSMICID is stored as float64 in DepMap (values like 905933.0).
    # Round and cast directly - regex-stripping the decimal would bake the ".0"
    # into the digits and produce 10x the correct IDs.
    df["COSMIC_ID"] = pd.to_numeric(df["COSMIC_ID"], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=["COSMIC_ID"]).copy()
    df["COSMIC_ID"] = df["COSMIC_ID"].astype(int)
    return df[["DepMap_ID", "COSMIC_ID"]].drop_duplicates(subset=["DepMap_ID"])


def _clean_expression_header(cols: list[str]) -> list[str]:
    """CCLE columns are like 'TSPAN6 (7105)' - strip the paren suffix to get symbol."""
    out = []
    for c in cols:
        sym = c.split("(")[0].strip()
        out.append(sym)
    return out


_CCLE_META_COLS = {"", "SequencingID", "ModelConditionID", "ModelID",
                   "IsDefaultEntryForMC", "IsDefaultEntryForModel", "Unnamed: 0"}


def _load_ccle_matrix(path: Path, genes_filter: list[str] | None) -> pd.DataFrame:
    """Load a CCLE expression/CNV file, filter to default entries, index by ModelID.
    Columns returned are cleaned gene symbols (paren-suffix stripped).
    If genes_filter is given, only those gene symbols are returned as columns.
    """
    # First pass: read just metadata cols to learn which rows to keep.
    meta = pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel"])
    keep_mask = meta["IsDefaultEntryForModel"].astype(str).str.strip().str.lower().eq("yes")
    keep_row_ix = meta.index[keep_mask].tolist()
    model_ids = meta.loc[keep_mask, "ModelID"].tolist()

    # Second pass: read full file (all columns). Memory-heavy but simpler than chunking.
    df = pd.read_csv(path)
    df = df.iloc[keep_row_ix].reset_index(drop=True)
    df.index = pd.Index(model_ids, name="DepMap_ID")

    meta_present = [c for c in df.columns if c in _CCLE_META_COLS or c.startswith("Unnamed")]
    gene_cols_raw = [c for c in df.columns if c not in meta_present]
    gene_cols_sym = _clean_expression_header(gene_cols_raw)
    df = df[gene_cols_raw].copy()
    df.columns = gene_cols_sym
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    if genes_filter is not None:
        kept = [g for g in genes_filter if g in df.columns]
        df = df[kept]
    return df


def _build_expression(model_map: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = _load_ccle_matrix(RAW / "ccle_expression.csv", genes_filter=genes)
    kept = df.columns.tolist()
    log.info("Expression genes matched: %d / %d", len(kept), len(genes))
    if len(kept) < 500:
        raise AssertionError(f"too few L1000 genes found in CCLE expression: {len(kept)}")
    expr = df.reset_index().merge(model_map, on="DepMap_ID", how="inner")
    expr = expr.set_index("COSMIC_ID")[kept]
    return expr, kept


def _build_mutations(model_map: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(RAW / "ccle_mutations.csv")
    id_col = "ModelID" if "ModelID" in df.columns else "DepMap_ID"
    gene_col = "HugoSymbol" if "HugoSymbol" in df.columns else "Hugo_Symbol"
    variant_col = "VariantInfo" if "VariantInfo" in df.columns else "Variant_Classification"
    # Keep non-silent mutations only
    if variant_col in df.columns:
        nonsilent = df[~df[variant_col].astype(str).str.contains("silent", case=False, na=False)].copy()
    else:
        nonsilent = df
    mut_counts = nonsilent[gene_col].value_counts()
    top_genes = mut_counts.head(top_n).index.tolist()
    # Binary matrix per (DepMap_ID, gene)
    sub = nonsilent[nonsilent[gene_col].isin(top_genes)][[id_col, gene_col]].drop_duplicates()
    sub["present"] = 1
    mat = sub.pivot(index=id_col, columns=gene_col, values="present").fillna(0).astype(int)
    # Ensure every top gene is present as a column
    for g in top_genes:
        if g not in mat.columns:
            mat[g] = 0
    mat = mat[top_genes]
    mat.index.name = "DepMap_ID"
    merged = mat.reset_index().merge(model_map, on="DepMap_ID", how="inner")
    return merged.set_index("COSMIC_ID")[top_genes], top_genes


def _build_cnv(model_map: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    df = _load_ccle_matrix(RAW / "ccle_cnv.csv", genes_filter=None)
    present = [g for g in genes if g in df.columns]
    cnv = df[present].copy()
    # Missing genes filled with 0 (log2 of diploid)
    for g in genes:
        if g not in cnv.columns:
            cnv[g] = 0.0
    cnv = cnv[genes]
    merged = cnv.reset_index().merge(model_map, on="DepMap_ID", how="inner")
    return merged.set_index("COSMIC_ID")[genes]


def build_cell_features(train_cosmic_ids: set[int] | None = None, top_mutation_genes: int = 200) -> dict:
    """Assemble 1378-d vectors per COSMIC_ID.

    train_cosmic_ids: if provided, StandardScaler is fit only on those cell lines.
                      If None, scaler is fit on ALL available cell lines (use only for debug).
    """
    genes = _load_l1000_genes()
    model_map = _load_depmap_model()
    log.info("DepMap ID<->COSMIC_ID pairs: %d", len(model_map))

    expr, expr_cols = _build_expression(model_map, genes)
    mut,  mut_cols  = _build_mutations(model_map, top_mutation_genes)
    cnv                = _build_cnv(model_map, mut_cols)

    # Align on intersection of COSMIC_IDs
    common = expr.index.intersection(mut.index).intersection(cnv.index)
    log.info("COSMIC_IDs with full feature set: %d", len(common))
    expr = expr.loc[common]
    mut  = mut.loc[common]
    cnv  = cnv.loc[common]

    # Column-name-suffixed so they're unique in the metadata list
    feature_names: list[str] = (
        [f"expr_{g}" for g in expr_cols]
        + [f"mut_{g}"  for g in mut_cols]
        + [f"cnv_{g}"  for g in mut_cols]
    )
    mat = np.concatenate([expr.values, mut.values, cnv.values], axis=1).astype(np.float32)
    n_feat = mat.shape[1]
    log.info("Raw feature matrix: %s (%d dims)", mat.shape, n_feat)

    # Fit scaler on training indices only
    scaler = StandardScaler()
    if train_cosmic_ids is None:
        log.warning("No train_cosmic_ids provided; scaler fit on ALL cell lines.")
        scaler.fit(mat)
    else:
        train_mask = np.array([c in train_cosmic_ids for c in common])
        if train_mask.sum() < 10:
            raise AssertionError(f"only {train_mask.sum()} train cell lines - too few to fit scaler")
        scaler.fit(mat[train_mask])
        log.info("Scaler fit on %d train cell lines (of %d total)", train_mask.sum(), len(common))
    mat = scaler.transform(mat).astype(np.float32)
    # Zero-variance columns -> NaN after division; replace with 0 (no predictive signal).
    nan_count = int(np.isnan(mat).sum())
    if nan_count:
        log.info("replaced %d NaN cells from zero-variance features with 0", nan_count)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    features = {int(c): mat[i] for i, c in enumerate(common)}

    PROC.mkdir(parents=True, exist_ok=True)
    with open(PROC / "cell_features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(PROC / "cell_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta_path = PROC / "feature_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["cell_features"] = feature_names
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("cell_features.pkl written: %d cell lines x %d dims", len(features), n_feat)
    return features
