import streamlit as st
import os
import requests
import pandas as pd
from typing import Optional

# ============================================
# CONFIGURATION
# ============================================
# Get API base URL from environment variable or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AutoPharmaX - AI-Powered Drug Efficacy Prediction",
    page_icon="💊",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
def local_css():
    st.markdown("""
        <style>
            /* General Body Styles */
            body, .main {
                background-color: #000000 !important;
                color: #FFFFFF;
                font-family: 'Helvetica', sans-serif;
            }

            /* Remove Streamlit Header and Footer */
            header, footer {
                visibility: hidden;
            }
            
            /* Custom Header */
            .custom-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 5%;
                width: 100%;
                background-color: #000000;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 1000;
            }
            .custom-header .logo {
                font-size: 2rem;
                font-weight: bold;
                color: #D4AF37; /* Gold-like color */
            }
            .custom-header .nav-links a {
                color: #FFFFFF;
                text-decoration: none;
                margin-left: 2rem;
                font-size: 1.1rem;
            }

            /* Main content padding to avoid overlap with custom header */
            .block-container {
                padding-top: 6rem !important;
            }

            /* Hero Section */
            .hero-section {
                text-align: center;
                padding: 4rem 2rem;
                background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('https://www.toptal.com/designers/subtlepatterns/uploads/double-bubble-outline.png');
                background-size: cover;
            }
            .hero-section h1 {
                font-size: 3.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            .hero-section p {
                font-size: 1.2rem;
                max-width: 800px;
                margin: auto;
                margin-bottom: 2rem;
            }

            /* Why Us Section */
            .why-us-section {
                text-align: center;
                padding: 4rem 2rem;
            }
            .why-us-section h2 {
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 3rem;
            }
            .feature-card {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 10px;
                padding: 2rem;
                text-align: left;
                position: relative;
                height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-bottom: 2rem;
            }
            .feature-card h3 {
                font-size: 1.5rem;
                font-weight: bold;
                color: #D4AF37;
                margin-bottom: 1rem;
            }
            .feature-card p {
                font-size: 1rem;
                color: #CCCCCC;
            }
            .feature-number {
                font-size: 8rem;
                font-weight: bold;
                color: #FFD700; /* Bright Gold */
                opacity: 0.3;
                position: absolute;
                bottom: -1rem;
                right: 1rem;
                line-height: 1;
            }

            /* Prediction Section */
            #predict {
                background-color: #111111;
                padding: 3rem;
                border-radius: 10px;
                border: 1px solid #333333;
                margin-top: 4rem;
            }
            
            /* Styling for selectbox and button */
            .stSelectbox > div > div {
                background-color: #333;
                color: white;
            }
            .stButton > button {
                border: 2px solid #D4AF37;
                background-color: #D4AF37;
                color: #000000;
                padding: 0.8rem 1.5rem;
                border-radius: 8px;
                font-size: 1.1rem;
                font-weight: bold;
                width: 100%;
            }
            .stButton > button:hover {
                background-color: transparent;
                color: #D4AF37;
            }

            /* Metric styling */
            .stMetric {
                background-color: #1a1a1a;
                border: 1px solid #333;
                padding: 1rem;
                border-radius: 8px;
            }
            .stMetric > label {
                color: #D4AF37;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 2rem;
                margin-top: 4rem;
                color: #555;
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ============================================
# API FUNCTIONS (same as before)
# ============================================
@st.cache_data(ttl=300)
def fetch_available_options():
    try:
        response = requests.get(f"{API_BASE_URL}/options", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch options from API: {e}")
        return None

def predict_ic50(cell_line: str, drug_name: str):
    try:
        payload = {"cell_line": cell_line, "drug_name": drug_name}
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        return None

# ============================================
# HEADER
# ============================================
st.markdown("""
    <div class="custom-header">
        <div class="logo">AutoPharmaX</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#predict">Predict</a>
            <a href="#contact">Contact</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================================
# HERO SECTION
# ============================================
st.markdown('<a id="home"></a>', unsafe_allow_html=True)
st.markdown("""
    <div class="hero-section">
        <h1>AI-Powered Drug Efficacy Prediction<br>for Personalized Oncology</h1>
        <p>Instantly predict drug response with high accuracy. Accelerate your research, reduce costs, and unlock the future of personalized medicine.</p>
        <a href="#predict" style="text-decoration: none;">
             <button style="border: 2px solid #D4AF37; background-color: transparent; color: #D4AF37; padding: 0.8rem 1.5rem; border-radius: 8px; font-size: 1.1rem; font-weight: bold; cursor: pointer;">
                Predict Now ↗
             </button>
        </a>
    </div>
""", unsafe_allow_html=True)

# ============================================
# WHY US? SECTION
# ============================================
st.markdown('<a id="about"></a>', unsafe_allow_html=True)
st.markdown("""
    <div class="why-us-section">
        <h2>Why Us?</h2>
    </div>
""", unsafe_allow_html=True)

features = {
    "1": {
        "title": "Unmatched Accuracy",
        "text": "Our fine-tuned XGBoost model achieves a 99.22% R² Score, ensuring predictions that closely mirror real-world lab outcomes."
    },
    "2": {
        "title": "High Correlation",
        "text": "We demonstrate a 99.62% Pearson correlation with actual experimental values, validating our model's predictive power."
    },
    "3": {
        "title": "Impressive Low Error",
        "text": "With a Root Mean Square Error (RMSE) of only 0.2512, our predictions are precise, reliable, and ready for critical research applications."
    },
    "4": {
        "title": "Production Ready & Scalable",
        "text": "AutoPharmaX is a fully deployed application, built on a robust pipeline ready to integrate into real-world research workflows and scale with your demands."
    },
    "5": {
        "title": "End-to-End Integrity",
        "text": "Our predictions are powered by a comprehensive data pipeline, from data merging and feature engineering to final deployment, ensuring quality, consistency, and traceability."
    },
    "6": {
        "title": "Best-in-Class Technology",
        "text": "To provide the best technology, we rigorously tested multiple models. Our tuned XGBoost was chosen for its demonstrably superior performance."
    }
}

cols = st.columns(3)
for i, (num, content) in enumerate(features.items()):
    if i >= 3: # Move to next row
        break
    with cols[i]:
        st.markdown(f"""
            <div class="feature-card">
                <div class="feature-number">{num}</div>
                <h3>{content['title']}</h3>
                <p>{content['text']}</p>
            </div>
        """, unsafe_allow_html=True)

cols = st.columns(3)
for i, (num, content) in enumerate(list(features.items())[3:]):
    with cols[i]:
        st.markdown(f"""
            <div class="feature-card">
                <div class="feature-number">{num}</div>
                <h3>{content['title']}</h3>
                <p>{content['text']}</p>
            </div>
        """, unsafe_allow_html=True)


# ============================================
# PREDICTION SECTION
# ============================================
st.markdown('<a id="predict"></a>', unsafe_allow_html=True)
with st.container():
    st.markdown("<h2 style='text-align: center; font-size: 3rem; margin-top: 5rem;'>Make a Prediction</h2>", unsafe_allow_html=True)
    
    options_data = fetch_available_options()

    if options_data is None:
        st.error("⚠️ Failed to load cell lines and drugs from API. Please ensure the backend is running.")
        st.stop()

    cell_lines = options_data.get("cell_lines", [])
    drugs = options_data.get("drugs", [])

    if not cell_lines or not drugs:
        st.error("⚠️ No cell lines or drugs available from the API.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_cell_line = st.selectbox("🧬 Select Cancer Cell Line", cell_lines)
    with col2:
        selected_drug = st.selectbox("💉 Select Drug", drugs)

    if st.button("🔬 Predict IC50"):
        with st.spinner("🔄 Making prediction..."):
            result = predict_ic50(selected_cell_line, selected_drug)
            if result:
                st.success("✅ Prediction Complete!")
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>📊 Prediction Results</h3>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Predicted LN(IC50)", f"{result['predicted_ln_ic50']:.4f}")
                res_col2.metric("Predicted IC50 (µM)", f"{result['predicted_ic50']:.4f}")

                if 'actual_ln_ic50' in result:
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center;'>📈 Comparison with Actual Value</h3>", unsafe_allow_html=True)
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    comp_col1.metric("Actual LN(IC50)", f"{result['actual_ln_ic50']:.4f}")
                    comp_col2.metric("Actual IC50 (µM)", f"{result['actual_ic50']:.4f}")
                    error = result.get('absolute_error', 0)
                    comp_col3.metric("Absolute Error", f"{error:.4f}", delta=f"{-error:.4f}", delta_color="inverse")

# ============================================
# FOOTER
# ============================================
st.markdown('<a id="contact"></a>', unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p>AutoPharmaX | AI for Personalized Medicine</p>
    </div>
""", unsafe_allow_html=True)
