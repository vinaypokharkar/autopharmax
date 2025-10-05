import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Drug Response Predictor",
    page_icon="💊",
    layout="centered"
)

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    """Load trained model, scaler, and metadata"""
    model_dir = "C:\\Users\\vinay\\OneDrive\\Desktop\\hackathon\\autopharma\\trained_models"
    
    try:
        # Load best model (note the parentheses in filename)
        model_path = os.path.join(model_dir, 'best_model_XGBoost_(Tuned).pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(f"Model directory: {model_dir}")
        st.error(f"Files in directory: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory not found'}")
        return None, None, None

@st.cache_data
def load_original_data():
    """Load original data to get available cell lines and drugs"""
    try:
        data = pd.read_csv("C:\\Users\\vinay\\OneDrive\\Desktop\\hackathon\\autopharma\\dataset\\merged_data.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ============================================
# LOAD EVERYTHING
# ============================================
model, scaler, metadata = load_models()
original_data = load_original_data()

# ============================================
# UI
# ============================================
st.title("💊 Drug Response Predictor")
st.markdown("---")

if model is None or original_data is None:
    st.error("⚠️ Failed to load models or data. Please check file paths.")
    st.stop()

# Extract unique values
cell_lines = sorted(original_data['TCGA_DESC'].unique())
drugs = sorted(original_data['DRUG_NAME'].unique()) if 'DRUG_NAME' in original_data.columns else ['Drug A', 'Drug B']

# ============================================
# INPUT SECTION
# ============================================
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
# PREDICTION BUTTON
# ============================================
if st.button("🔬 Predict IC50", use_container_width=True, type="primary"):
    
    with st.spinner("Predicting..."):
        
        try:
            # Filter data for selected combination
            filtered_data = original_data[
                (original_data['TCGA_DESC'] == selected_cell_line) &
                (original_data['DRUG_NAME'] == selected_drug)
            ]
            
            if len(filtered_data) == 0:
                st.warning("⚠️ No data available for this combination. Using average features.")
                # Use mean values for features
                feature_data = original_data[metadata['feature_columns']].mean().to_frame().T
            else:
                # Get the first matching row's features
                feature_data = filtered_data[metadata['feature_columns']].iloc[0:1]
            
            # Scale features
            X_scaled = scaler.transform(feature_data)
            
            # Predict
            ln_ic50_pred = model.predict(X_scaled)[0]
            ic50_pred = np.exp(ln_ic50_pred)  # Convert from log scale
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("---")
            
            # Results in columns
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    label="Predicted LN(IC50)",
                    value=f"{ln_ic50_pred:.4f}"
                )
            
            with res_col2:
                st.metric(
                    label="Predicted IC50 (µM)",
                    value=f"{ic50_pred:.4f}"
                )
            
            # Additional info
            if len(filtered_data) > 0 and 'LN_IC50' in filtered_data.columns:
                actual_ln_ic50 = filtered_data['LN_IC50'].iloc[0]
                actual_ic50 = np.exp(actual_ln_ic50)
                
                st.markdown("---")
                st.markdown("#### 📊 Comparison with Actual Value")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric("Actual LN(IC50)", f"{actual_ln_ic50:.4f}")
                
                with comp_col2:
                    st.metric("Actual IC50 (µM)", f"{actual_ic50:.4f}")
                
                with comp_col3:
                    error = abs(ln_ic50_pred - actual_ln_ic50)
                    st.metric("Absolute Error", f"{error:.4f}")
            
            # Model info
            with st.expander("ℹ️ Model Information"):
                st.write(f"**Model:** {metadata['best_model_name']}")
                st.write(f"**Training Date:** {metadata['training_date']}")
                st.write(f"**Number of Features:** {len(metadata['feature_columns'])}")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🧪 Drug Response Prediction System | Powered by Machine Learning")