"""Stage 1 - Data ingestion.

Orchestrates these independent sub-steps with granular skip flags:
  1. copy_gdsc2            (local dataset/ -> data/raw/)
  2. copy_cell_lines_details (local dataset/ -> data/raw/)
  3. download L1000 gene list  (S3, small, usually works)
  4. download DepMap CCLE files (big, may be blocked - --skip-depmap)
  5. download DrugComb summary  (may be blocked - --skip-drugcomb)
  6. fetch SMILES from PubChem  (usually works)
  7. validate every raw file present
  8. dvc add data/raw/ (best-effort)

If --skip-depmap or --skip-drugcomb is set, stage 1 trusts that the
user has manually placed the files at the expected paths and only
validates what's present.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import download, validate  # noqa: E402

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-depmap", action="store_true",
                    help="Skip CCLE / DepMap Model.csv download (you placed them manually)")
    ap.add_argument("--skip-drugcomb", action="store_true",
                    help="Skip DrugComb download (you placed it manually)")
    ap.add_argument("--skip-l1000", action="store_true",
                    help="Skip L1000 gene-list download")
    ap.add_argument("--skip-pubchem", action="store_true",
                    help="Skip PubChem SMILES fetch (you placed drug_smiles.csv manually)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Skip final validate.run_all() - useful during partial test runs")
    ap.add_argument("--skip-dvc", action="store_true",
                    help="Skip `dvc add data/raw/`")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files exist")
    ap.add_argument("--debug", action="store_true",
                    help="PubChem: only first 10 drugs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("[1/7] copy GDSC2 + Cell_Lines_Details from dataset/")
    gdsc_path = download.copy_gdsc2(force=args.force)
    download.copy_cell_lines_details(force=args.force)

    if not args.skip_l1000:
        log.info("[2/7] download L1000 landmark gene list")
        if not download.download_l1000_genes(force=args.force):
            log.error("L1000 download failed. URL: %s", download.L1000_URL)
            return 2
    else:
        log.info("[2/7] --skip-l1000")

    if not args.skip_depmap:
        log.info("[3/7] download DepMap (CCLE expression/mutations/CNV + Model.csv)")
        results = download.download_depmap(force=args.force)
        failed = [k for k, ok in results.items() if not ok]
        if failed:
            log.error("DepMap downloads failed: %s", failed)
            log.error("-> Download manually per DOWNLOAD_INSTRUCTIONS.md and rerun with --skip-depmap")
            return 2
    else:
        log.info("[3/7] --skip-depmap (trusting manual placement)")

    if not args.skip_drugcomb:
        log.info("[4/7] download DrugComb summary")
        if not download.download_drugcomb(force=args.force):
            log.error("DrugComb download failed. URL: %s", download.DRUGCOMB_URL)
            log.error("-> Download manually per DOWNLOAD_INSTRUCTIONS.md and rerun with --skip-drugcomb")
            return 2
    else:
        log.info("[4/7] --skip-drugcomb (trusting manual placement)")

    if not args.skip_pubchem:
        log.info("[5/7] fetch PubChem SMILES for GDSC2 drugs")
        gdsc = pd.read_csv(gdsc_path, usecols=["DRUG_NAME"])
        drugs = gdsc["DRUG_NAME"].dropna().unique().tolist()
        if args.debug:
            drugs = drugs[:10]
            log.info("--debug: only %d drugs", len(drugs))
        download.download_drug_smiles(drugs, force=args.force)
    else:
        log.info("[5/7] --skip-pubchem")

    if not args.skip_validate:
        log.info("[6/7] validate raw data")
        try:
            validate.run_all()
        except AssertionError as e:
            log.error("VALIDATION FAILED: %s", e)
            return 3
    else:
        log.info("[6/7] --skip-validate")

    if not args.skip_dvc:
        log.info("[7/7] dvc add data/raw")
        try:
            subprocess.run(["dvc", "add", "data/raw"], cwd=ROOT, check=True)
            log.info("dvc add data/raw -> tracked")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning("dvc add skipped (%s). Run `dvc init && dvc add data/raw` manually.", e)
    else:
        log.info("[7/7] --skip-dvc")

    log.info("Stage 1 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
