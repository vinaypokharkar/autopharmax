# Manual data download instructions

Your machine currently can't reach `drugcomb.fimm.fi` and the DepMap portal's
direct-download endpoint (portal returns 403, DrugComb domain is
firewall-blocked). Download the 5 files below in a browser and drop them
at the exact target paths.

After placement, run:

```bash
.venv/Scripts/python.exe src/pipeline/stage1_ingest.py --skip-depmap --skip-drugcomb
```

Stage 1 will still run the L1000 gene list download, PubChem SMILES fetch,
validation, and DVC-add.

---

## DepMap (CCLE expression / mutations / CNV + ID map)

Open https://depmap.org/portal/data_page/?tab=allData in a browser, pick
the **latest DepMap Public release** (as of today that's 25Q4 or 26Q1),
and download these four files from that release page:

| Source filename (DepMap portal)                       | Save to                      | Approx size |
|-------------------------------------------------------|------------------------------|------------:|
| `Model.csv`                                           | `data/raw/depmap_model.csv`  | ~500 KB     |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv`       | `data/raw/ccle_expression.csv`| ~200 MB    |
| `OmicsSomaticMutations.csv`                           | `data/raw/ccle_mutations.csv`| ~150 MB     |
| `OmicsCNGene.csv`                                     | `data/raw/ccle_cnv.csv`      | ~150 MB     |

> `Model.csv` is the master ID table. It must contain both `ModelID`
> (DepMap_ID) and a `COSMICID` column so we can cross-walk GDSC2 IDs to
> CCLE IDs. Older releases used different column names — `cell_features.py`
> auto-handles `DepMap_ID`/`ModelID` and `COSMIC_ID`/`COSMICID`.

### Sanity check after download

```bash
head -1 data/raw/ccle_expression.csv   # should show a header with gene symbols like "TSPAN6 (7105)"
head -1 data/raw/depmap_model.csv       # should include ModelID and COSMICID
wc -l data/raw/ccle_*.csv               # each should have >1000 lines
```

---

## DrugComb (synergy labels)

Open https://drugcomb.fimm.fi/ in a browser. Under the **Downloads**
section, grab:

| Source filename       | Save to                 | Approx size |
|-----------------------|-------------------------|------------:|
| `summary_v_1_5.csv`   | `data/raw/drugcomb.csv` | ~200 MB     |

Required columns: `drug_row`, `drug_col`, `cell_line_name`, `synergy_loewe`.

> If `drugcomb.fimm.fi` is slow / flaky, a common mirror is Kaggle search
> "DrugComb summary v1.5" — any copy with those four columns works.

---

## Troubleshooting

If you hit "Invalid Cross-Origin" or "403 Forbidden" on DepMap when
browser-downloading, log into a DepMap account (free, any email works —
they just gate the download API).

If `stage1_ingest.py --skip-depmap --skip-drugcomb` still fails at the
validate step, it will tell you which specific file is missing or has a
wrong schema. Fix that file and rerun.
