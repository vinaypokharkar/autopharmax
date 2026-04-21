"""AutoPharmaX Streamlit UI.

Three pages via sidebar nav:
  1) Single-drug IC50 prediction with SHAP attribution bars
  2) Drug-combination synergy, with a "find best partner" scan
  3) Model performance card (4 split metrics, scatter plots)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AutoPharmaX", layout="wide", page_icon="🧬")


# ---------- API helpers --------------------------------------------------

def _api_get(path: str, **kw):
    r = requests.get(f"{API_BASE}{path}", timeout=20, **kw)
    r.raise_for_status()
    return r.json()


def _api_post(path: str, payload: dict):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=600)
def fetch_health():
    return _api_get("/health")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_cell_lines():
    return _api_get("/api/v1/cell-lines")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_drugs():
    return _api_get("/api/v1/drugs")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_metrics():
    return _api_get("/api/v1/metrics")


# ---------- Shared cell-line / drug selectors ----------------------------

def _cell_selector(label: str = "Cancer cell line") -> int | None:
    cells = fetch_cell_lines()
    if not cells:
        st.error("No cell lines available from API")
        return None
    labels = []
    id_by_label = {}
    for c in cells:
        tag = f"[{c['tissue']}] " if c.get("tissue") else ""
        lab = f"{c.get('cell_line_name') or '?'} {tag}(COSMIC {c['cosmic_id']})"
        labels.append(lab)
        id_by_label[lab] = c["cosmic_id"]
    pick = st.selectbox(label, labels, index=0)
    return id_by_label[pick]


def _drug_selector(label: str, key: str) -> str | None:
    drugs = fetch_drugs()
    if not drugs:
        st.error("No drugs available from API")
        return None
    return st.selectbox(label, drugs, key=key)


# ---------- Page 1: single-drug prediction -------------------------------

def page_single():
    st.title("🧬 Single-drug IC50 prediction")
    st.caption(
        "Predict LN(IC50) for a drug-cell line pair from the shared cell omics + "
        "molecular-fingerprint model. A **novel prediction** here is a feature, "
        "not a warning: it means the pair was not in training data, and the model "
        "is genuinely generalising from molecular biology."
    )

    col1, col2 = st.columns([2, 2])
    with col1:
        cosmic = _cell_selector()
    with col2:
        drug = _drug_selector("Drug", "single_drug")

    if st.button("Predict IC50", type="primary", disabled=(cosmic is None or drug is None)):
        try:
            with st.spinner("Running model on GPU..."):
                resp = _api_post("/api/v1/predict/single",
                                 {"cosmic_id": int(cosmic), "drug_name": drug})
        except Exception as e:
            st.error(str(e)); return

        # Header badge + core numbers
        is_novel = bool(resp["is_novel_prediction"])
        badge = ("🟢  **Novel prediction** — this cell-line + drug pair was not in "
                 "training data. The model is genuinely predicting from molecular biology.")
        if not is_novel:
            badge = "⚪  Known pair (the actual value is below, for reference only — "\
                    "the model was NOT given it at inference time)."
        st.markdown(badge)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted LN(IC50)", f"{resp['predicted_ln_ic50']:.3f}")
        m2.metric("Predicted IC50 (µM)", f"{resp['predicted_ic50_uM']:.3f}")
        m3.metric("Sensitivity class",   resp["sensitivity_class"])

        if resp.get("actual_ln_ic50") is not None:
            a, e = st.columns(2)
            a.metric("Actual LN(IC50) (reference)", f"{resp['actual_ln_ic50']:.3f}")
            e.metric("Absolute error",              f"{resp['absolute_error']:.3f}")

        st.divider()
        st.subheader("Feature attributions")
        sh1, sh2 = st.columns(2)
        with sh1:
            st.caption("Top cell-line features (mean |SHAP| across test set)")
            df = pd.DataFrame(resp["top_cell_features"])
            st.bar_chart(df.set_index("feature")["importance"], horizontal=True)
            st.caption(
                "SHAP measures how much each feature moved the prediction away "
                "from the dataset average. `expr_X` is gene expression, `mut_X` "
                "is a binary mutation flag, `cnv_X` is copy-number log2 ratio."
            )
        with sh2:
            st.caption("Top drug descriptors (mean |SHAP|)")
            df = pd.DataFrame(resp["top_drug_features"])
            st.bar_chart(df.set_index("feature")["importance"], horizontal=True)
            st.caption(
                "Molecular descriptors from RDKit: physicochemical properties "
                "like aromatic ring count, H-bond donors/acceptors, topology. "
                "Morgan fingerprint bits are in the model but excluded from "
                "this view — individual bits aren't interpretable by name."
            )


# ---------- Page 2: combination synergy ----------------------------------

def page_combination():
    st.title("💊 Drug combination synergy")
    st.caption(
        "Predicted Loewe synergy score for two drugs on a cancer cell line. "
        "Score > 10 = synergistic, -10..10 = additive, < -10 = antagonistic."
    )

    cosmic = _cell_selector()
    col1, col2 = st.columns(2)
    with col1:
        drug_a = _drug_selector("Drug A", "comb_a")
    with col2:
        drug_b = _drug_selector("Drug B", "comb_b")

    ok = cosmic is not None and drug_a and drug_b and drug_a != drug_b
    if st.button("Predict synergy", type="primary", disabled=not ok):
        try:
            with st.spinner("Running synergy model..."):
                resp = _api_post("/api/v1/predict/combination",
                                 {"cosmic_id": int(cosmic),
                                  "drug_a": drug_a, "drug_b": drug_b})
        except Exception as e:
            st.error(str(e)); return

        score = resp["synergy_score"]
        cls   = resp["synergy_class"]
        if cls == "synergistic":
            st.success(f"Synergistic — Loewe score = **{score:.2f}**")
        elif cls == "antagonistic":
            st.error(f"Antagonistic — Loewe score = **{score:.2f}**")
        else:
            st.info(f"Additive — Loewe score = **{score:.2f}**")

        a, b = st.columns(2)
        a.metric(f"{drug_a} solo LN(IC50)", f"{resp['drug_a_ln_ic50']:.3f}")
        b.metric(f"{drug_b} solo LN(IC50)", f"{resp['drug_b_ln_ic50']:.3f}")

    st.divider()
    st.subheader("Find the best partner drug")
    st.caption(
        "For the chosen cell line + Drug A, rank every drug in the catalogue by "
        "predicted synergy and show the top 10. This is the flagship feature."
    )
    ok_scan = cosmic is not None and drug_a is not None
    if st.button("Find best partners for Drug A", disabled=not ok_scan):
        try:
            with st.spinner("Scanning ~230 drugs..."):
                resp = _api_post("/api/v1/predict/combination/scan",
                                 {"cosmic_id": int(cosmic),
                                  "drug_a": drug_a, "top_k": 10})
        except Exception as e:
            st.error(str(e)); return
        df = pd.DataFrame(resp["results"])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------- Page 3: model performance ------------------------------------

def page_metrics():
    st.title("📈 Model performance")

    try:
        m = fetch_metrics()
    except Exception as e:
        st.error(f"metrics unavailable: {e}"); return

    def _r2(d): return d.get("r2", float("nan"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Standard R²",           f"{_r2(m.get('standard', {})):.3f}")
    c2.metric("LDO R² (headline)",     f"{_r2(m.get('ldo', {})):.3f}")
    c3.metric("LCLO R²",               f"{_r2(m.get('lclo', {})):.3f}")
    c4.metric("Synergy LPO R²",        f"{_r2(m.get('lpo', {})):.3f}")

    st.caption(
        "The headline metric is **Leave-Drug-Out R²**: the test set contains "
        "drugs the model has NEVER seen. This is the honest evaluation of a "
        "drug-efficacy predictor — random-split R² (the 'Standard' column) "
        "overstates performance because it lets the model memorise pair-level "
        "noise."
    )

    st.divider()
    st.subheader("Predicted vs. actual")
    eval_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "eval"
    cols = st.columns(2)
    for i, name in enumerate(["standard", "ldo", "lclo", "lpo"]):
        path = eval_dir / f"predicted_vs_actual_{name}.png"
        with cols[i % 2]:
            if path.exists():
                st.image(str(path), caption=f"{name.upper()} test", use_container_width=True)
            else:
                st.caption(f"{path.name} not found - run stage 4 first.")

    st.divider()
    st.subheader("Full metrics")
    rows = []
    for split in ("standard", "ldo", "lclo", "lpo"):
        md = m.get(split, {})
        rows.append({
            "split": split,
            "n": md.get("n"),
            "R²": md.get("r2"),
            "Pearson": md.get("pearson"),
            "RMSE": md.get("rmse"),
            "MAE":  md.get("mae"),
            "AUROC (>10)": md.get("auroc"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------- Nav ----------------------------------------------------------

def main():
    st.sidebar.title("AutoPharmaX")
    try:
        h = fetch_health()
        st.sidebar.caption(f"API: `{API_BASE}`")
        st.sidebar.caption(f"Model: {h.get('model_version') or 'n/a'}")
        st.sidebar.caption(f"LDO R²: {h.get('single_drug_ldo_r2') or 'n/a':.3f}"
                           if h.get('single_drug_ldo_r2') is not None else "LDO R²: n/a")
        st.sidebar.caption(f"Device: {h.get('device')}")
        st.sidebar.caption(f"Cells: {h.get('cell_features')} · Drugs: {h.get('drug_features')}")
    except Exception as e:
        st.sidebar.error(f"API unreachable: {e}")

    page = st.sidebar.radio("Page", ["Single drug", "Combination", "Model performance"])
    if page == "Single drug":
        page_single()
    elif page == "Combination":
        page_combination()
    else:
        page_metrics()


if __name__ == "__main__":
    main()
