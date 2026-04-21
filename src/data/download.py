"""Download raw data sources for AutoPharmaX.

All files land in data/raw/. Existing files are skipped unless --force.
If a remote source fails, the function logs the expected URL and returns
False so the caller can decide whether to continue.

URLs are constants at the top so they can be overridden in one place if
a data release changes. CCLE/DepMap releases rotate - update RELEASE_TAG.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DATASET = ROOT / "dataset"

# ---- Static URLs (override via env if they rotate) ----------------------
# DepMap 24Q4 figshare direct-download file IDs.
DEPMAP_URLS = {
    "ccle_expression.csv":  "https://figshare.com/ndownloader/files/50615752",
    "ccle_mutations.csv":   "https://figshare.com/ndownloader/files/50615716",
    "ccle_cnv.csv":         "https://figshare.com/ndownloader/files/50615722",
    "depmap_model.csv":     "https://figshare.com/ndownloader/files/50615784",
}

DRUGCOMB_URL = "https://drugcomb.fimm.fi/download/summary_v_1_5.csv"

# L1000 landmark gene list. Source: cmap/LINCS2020 on S3.
# The file has 12k rows; filter feature_space == "landmark" -> 978 symbols.
L1000_URL = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt"

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
PUBCHEM_SLEEP = 0.25   # seconds between requests; PubChem ceiling is 5 req/s
PUBCHEM_RETRIES = 3


# ---- Helpers ------------------------------------------------------------

def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Stream a URL to dest with a progress bar. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0) or 0)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f, tqdm(
                desc=desc or dest.name, total=total, unit="B", unit_scale=True, disable=total == 0
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    f.write(chunk)
                    bar.update(len(chunk))
            tmp.replace(dest)
        return True
    except Exception as e:
        log.error("download failed %s -> %s: %s", url, dest.name, e)
        return False


def _skip_if_exists(dest: Path, force: bool) -> bool:
    if dest.exists() and not force:
        log.info("exists, skipping: %s", dest)
        return True
    return False


# ---- GDSC2 (local reuse) ------------------------------------------------

def copy_gdsc2(force: bool = False) -> Path:
    """Copy the existing GDSC2 CSV from dataset/ into data/raw/gdsc2.csv."""
    src = DATASET / "GDSC2-dataset.csv"
    dest = RAW / "gdsc2.csv"
    if not src.exists():
        raise FileNotFoundError(f"expected {src} (original GDSC2 download)")
    if _skip_if_exists(dest, force):
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log.info("copied %s -> %s", src, dest)
    return dest


def copy_cell_lines_details(force: bool = False) -> Path:
    src = DATASET / "Cell_Lines_Details.xlsx"
    dest = RAW / "cell_lines_details.xlsx"
    if not src.exists():
        raise FileNotFoundError(f"expected {src}")
    if _skip_if_exists(dest, force):
        return dest
    shutil.copy2(src, dest)
    log.info("copied %s -> %s", src, dest)
    return dest


# ---- DepMap / CCLE ------------------------------------------------------

def download_depmap(force: bool = False) -> dict[str, bool]:
    """Download CCLE expression / mutations / CNV + DepMap Model.csv."""
    results: dict[str, bool] = {}
    for name, url in DEPMAP_URLS.items():
        dest = RAW / name
        if _skip_if_exists(dest, force):
            results[name] = True
            continue
        results[name] = _download(url, dest, desc=name)
    return results


# ---- DrugComb -----------------------------------------------------------

def download_drugcomb(force: bool = False) -> bool:
    dest = RAW / "drugcomb.csv"
    if _skip_if_exists(dest, force):
        return True
    return _download(DRUGCOMB_URL, dest, desc="drugcomb")


# ---- L1000 landmark genes ----------------------------------------------

def download_l1000_genes(force: bool = False) -> bool:
    """Save 978 landmark gene symbols to data/raw/l1000_genes.txt.

    Source columns: gene_id, gene_symbol, ensembl_id, gene_title, gene_type, src, feature_space.
    We keep rows where feature_space == "landmark".
    """
    dest = RAW / "l1000_genes.txt"
    if _skip_if_exists(dest, force):
        return True
    tmp = RAW / "geneinfo_beta.txt"
    if not _download(L1000_URL, tmp, desc="L1000 genes"):
        return False
    symbols: list[str] = []
    with open(tmp, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        sym_idx = header.index("gene_symbol") if "gene_symbol" in header else 1
        fs_idx  = header.index("feature_space") if "feature_space" in header else -1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if fs_idx >= 0 and len(parts) > fs_idx and parts[fs_idx].strip() != "landmark":
                continue
            if len(parts) > sym_idx:
                symbols.append(parts[sym_idx].strip())
    symbols = [s for s in symbols if s]
    if len(symbols) < 500:
        log.error("L1000 parse: only %d landmark symbols found (expected ~978)", len(symbols))
        return False
    dest.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    tmp.unlink(missing_ok=True)
    log.info("L1000 genes saved: %d symbols", len(symbols))
    return True


# ---- PubChem SMILES -----------------------------------------------------

def _pubchem_smiles(name: str) -> str | None:
    # PubChem's property field is `SMILES` in the current API (the spec said
    # IsomericSMILES, but that returns empty). `SMILES` is the default isomeric form.
    url = f"{PUBCHEM_BASE}/{requests.utils.quote(name)}/property/SMILES/JSON"
    delay = PUBCHEM_SLEEP
    for _ in range(PUBCHEM_RETRIES):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                if not props:
                    return None
                p = props[0]
                return p.get("SMILES") or p.get("IsomericSMILES") or p.get("CanonicalSMILES")
            if r.status_code == 404:
                return None
            time.sleep(delay); delay *= 2
        except requests.RequestException:
            time.sleep(delay); delay *= 2
    return None


def download_drug_smiles(drug_names: Iterable[str], force: bool = False) -> Path:
    """Fetch IsomericSMILES for each drug name. Writes data/raw/drug_smiles.csv.

    Drugs that PubChem cannot resolve are appended to smiles_failures.txt -
    they are dropped downstream, never imputed.
    """
    dest = RAW / "drug_smiles.csv"
    failures_path = RAW / "smiles_failures.txt"
    have: dict[str, str] = {}
    if dest.exists() and not force:
        with open(dest, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split(",", 1)
                if len(parts) == 2:
                    have[parts[0]] = parts[1]

    drugs = sorted(set(drug_names))
    missing = [d for d in drugs if d not in have]
    if not missing:
        log.info("SMILES cache complete: %d drugs", len(have))
        return dest

    failures: list[str] = []
    for d in tqdm(missing, desc="PubChem SMILES"):
        smi = _pubchem_smiles(d)
        if smi:
            have[d] = smi
        else:
            failures.append(d)
        time.sleep(PUBCHEM_SLEEP)

    # Rewrite file from the merged cache
    RAW.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write("drug_name,smiles\n")
        for name in sorted(have):
            smi = have[name].replace(",", "")   # defensive CSV-safety
            f.write(f"{name},{smi}\n")

    if failures:
        with open(failures_path, "a", encoding="utf-8") as f:
            for name in failures:
                f.write(name + "\n")
        log.warning("SMILES failures: %d drugs (appended to %s)", len(failures), failures_path.name)

    log.info("SMILES resolved: %d / %d", len(have), len(drugs))
    return dest
