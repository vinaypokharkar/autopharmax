import streamlit as st
import requests
import pandas as pd
from typing import Optional

# ============================================
# CONFIGURATION
# ============================================
API_BASE_URL = "http://backend:8000/api/v1"  # Change this if your API is hosted elsewhere

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Drug Response Predictor",
    page_icon="💊",
    layout="centered"
)

# ============================================
# API FUNCTIONS
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_available_options():
    """Fetch available cell lines and drugs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/options", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch options from API: {e}")
        return None

def predict_ic50(cell_line: str, drug_name: str):
    """Make IC50 prediction via API"""
    try:
        payload = {
            "cell_line": cell_line,
            "drug_name": drug_name
        }
        response = requests.post(
            f"{API_BASE_URL}/predict", 
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        return None

def check_api_health():
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

# ============================================
# UI HEADER
# ============================================
st.title("💊 Drug Response Predictor")
st.markdown("---")

# ============================================
# API HEALTH CHECK
# ============================================
health_status = check_api_health()

if health_status is None:
    st.error("⚠️ Unable to connect to the API server. Please make sure the FastAPI backend is running.")
    st.info("💡 Start the backend with: `uvicorn main:app --reload`")
    st.stop()
elif health_status.get("status") != "healthy":
    st.warning("⚠️ API is running but models/data are not loaded properly.")
    st.json(health_status)
    st.stop()
else:
    st.success("✅ Connected to API successfully")

# ============================================
# LOAD OPTIONS
# ============================================
options_data = fetch_available_options()

if options_data is None:
    st.error("⚠️ Failed to load cell lines and drugs from API.")
    st.stop()

cell_lines = options_data.get("cell_lines", [])
drugs = options_data.get("drugs", [])

if not cell_lines or not drugs:
    st.error("⚠️ No cell lines or drugs available.")
    st.stop()

# ============================================
# INPUT SECTION
# ============================================
st.markdown("### 🔬 Select Parameters")

col1, col2 = st.columns(2)

with col1:
    selected_cell_line = st.selectbox(
        "🧬 Select Cancer Cell Line",
        cell_lines,
        help="Choose the cancer cell line type"
    )

with col2:
    selected_drug = st.selectbox(
        "💉 Select Drug",
        drugs,
        help="Choose the drug compound"
    )

st.markdown("---")

# ============================================
# PREDICTION SECTION
# ============================================
if st.button("🔬 Predict IC50", use_container_width=True, type="primary"):
    
    with st.spinner("🔄 Making prediction..."):
        
        # Call API
        result = predict_ic50(selected_cell_line, selected_drug)
        
        if result is not None:
            # Display success message
            st.success("✅ Prediction Complete!")
            
            st.markdown("---")
            
            # Display prediction results
            st.markdown("### 📊 Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    label="Predicted LN(IC50)",
                    value=f"{result['predicted_ln_ic50']:.4f}"
                )
            
            with res_col2:
                st.metric(
                    label="Predicted IC50 (µM)",
                    value=f"{result['predicted_ic50']:.4f}"
                )
            
            # Show actual values if available
            if result.get('actual_ln_ic50') is not None:
                st.markdown("---")
                st.markdown("### 📈 Comparison with Actual Value")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric(
                        "Actual LN(IC50)", 
                        f"{result['actual_ln_ic50']:.4f}"
                    )
                
                with comp_col2:
                    st.metric(
                        "Actual IC50 (µM)", 
                        f"{result['actual_ic50']:.4f}"
                    )
                
                with comp_col3:
                    error = result.get('absolute_error', 0)
                    st.metric(
                        "Absolute Error", 
                        f"{error:.4f}",
                        delta=f"{-error:.4f}",
                        delta_color="inverse"
                    )
                
                # Visualization of comparison
                comparison_df = pd.DataFrame({
                    'Type': ['Predicted', 'Actual'],
                    'LN(IC50)': [result['predicted_ln_ic50'], result['actual_ln_ic50']],
                    'IC50 (µM)': [result['predicted_ic50'], result['actual_ic50']]
                })
                
                st.markdown("---")
                st.markdown("### 📉 Visual Comparison")
                
                tab1, tab2 = st.tabs(["LN(IC50)", "IC50 (µM)"])
                
                with tab1:
                    st.bar_chart(
                        comparison_df.set_index('Type')['LN(IC50)'],
                        use_container_width=True
                    )
                
                with tab2:
                    st.bar_chart(
                        comparison_df.set_index('Type')['IC50 (µM)'],
                        use_container_width=True
                    )
            else:
                st.info("ℹ️ No actual value available for this combination (using averaged features)")
            
            # Model Information
            with st.expander("ℹ️ Model Information"):
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Model:** {result['model_name']}")
                    st.write(f"**Number of Features:** {result['num_features']}")
                
                with info_col2:
                    st.write(f"**Training Date:** {result['training_date']}")
                    st.write(f"**Cell Line:** {selected_cell_line}")
                    st.write(f"**Drug:** {selected_drug}")

# ============================================
# SIDEBAR - Additional Info
# ============================================
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    This application predicts IC50 values (drug effectiveness) for cancer cell lines.
    
    **How it works:**
    1. Select a cancer cell line
    2. Choose a drug compound
    3. Get predicted IC50 values
    
    **What is IC50?**
    IC50 is the concentration of a drug needed to inhibit 50% of cell growth.
    Lower values indicate higher drug effectiveness.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔗 API Status")
    if health_status:
        st.success("✅ API Connected")
        st.text(f"Model Loaded: {health_status.get('model_loaded', False)}")
        st.text(f"Data Loaded: {health_status.get('data_loaded', False)}")
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Info")
    if options_data:
        st.metric("Cell Lines", len(cell_lines))
        st.metric("Drugs", len(drugs))
        st.metric("Total Combinations", len(cell_lines) * len(drugs))

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🧪 Drug Response Prediction System | Powered by Machine Learning & FastAPI")