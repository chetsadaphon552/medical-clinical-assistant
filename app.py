import streamlit as st
import os
import time
from src.agent import MedicalSymptomAssistant

# Page Configuration
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="⚕️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #F4F7F6;
    }
    
    /* Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1A365D;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Force all Markdown text to be BLACK/Dark Blue */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #2D3748 !important;
    }
    
    /* Style the table */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        border: 1px solid #CBD5E0;
    }
    th {
        background-color: #E2E8F0;
        color: #2D3748;
        font-weight: bold;
        text-align: left;
        padding: 12px;
        border: 1px solid #CBD5E0;
    }
    td {
        padding: 12px;
        border: 1px solid #CBD5E0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A365D !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'agent' not in st.session_state:
    with st.spinner("Initializing System..."):
        try:
            st.session_state.agent = MedicalSymptomAssistant()
        except Exception as e:
            st.error(f"Error: {e}")

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>CDSS Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Clear History"):
        if os.path.exists("logs/agent.log"):
            with open("logs/agent.log", "w") as f: f.write("")
        st.rerun()

# --- Header Section ---
st.title("Clinical Decision Support System")
st.markdown("ระบบสนับสนุนการตัดสินใจทางคลินิก (CDSS)")
st.markdown("---")

# Layout
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown("### Patient Clinical Data")
    user_input = st.text_area("Chief Complaint / Symptoms:", placeholder="ระบุอาการแสดงทางคลินิกของผู้ป่วย...", height=150)
    analyze_btn = st.button("Analyze Symptoms")

with col_right:
    if analyze_btn and user_input:
        with st.spinner("Processing Clinical Data..."):
            try:
                response = st.session_state.agent.query(user_input)
                
                st.markdown("### Clinical Analysis Report")
                # Use a container for the report with a border
                with st.container(border=True):
                    st.markdown(response) # Pure markdown for correct table rendering
                
                with st.expander("System Trace Logs"):
                    if os.path.exists("logs/agent.log"):
                        with open("logs/agent.log", "r", encoding='utf-8') as f:
                            st.code("".join(f.readlines()[-20:]), language="text")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Please input patient symptoms on the left panel to begin the analysis.")
