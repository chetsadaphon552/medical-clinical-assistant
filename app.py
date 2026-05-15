import streamlit as st
import os
import time
from src.agent import MedicalSymptomAssistant

# Page Configuration
st.set_page_config(
    page_title="Clinical CDS Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Force all Markdown text to be BLACK */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #000000 !important;
    }
    
    /* Style the table */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        border: 1px solid #e2e8f0;
    }
    th {
        background-color: #f1f5f9;
        font-weight: bold;
        text-align: left;
        padding: 12px;
        border: 1px solid #e2e8f0;
    }
    td {
        padding: 12px;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'agent' not in st.session_state:
    with st.spinner("🏥 Starting Agent..."):
        try:
            st.session_state.agent = MedicalSymptomAssistant()
        except Exception as e:
            st.error(f"Error: {e}")

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🏥 CDS Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ ล้างประวัติ"):
        if os.path.exists("logs/agent.log"):
            with open("logs/agent.log", "w") as f: f.write("")
        st.rerun()

# --- Header Section ---
st.title("🏥 Clinical Decision Assistant")
st.markdown("ระบบสนับสนุนการตัดสินใจทางคลินิกอัจฉริยะ (Professional CDSS)")

# --- DEBUG SECTION (ตรวจสอบไฟล์) ---
if st.checkbox("🔍 ตรวจสอบความพร้อมของระบบ (Debug)"):
    import os
    st.write("Current Directory:", os.getcwd())
    target_path = "data/vectordb/faiss_index"
    if os.path.exists(target_path):
        st.success(f"✅ โฟลเดอร์ {target_path} มีอยู่จริง")
        st.write("รายชื่อไฟล์ในโฟลเดอร์:", os.listdir(target_path))
    else:
        st.error(f"❌ ไม่พบโฟลเดอร์ {target_path}")
        # ลองหาดูว่ามีโฟลเดอร์ไหนใกล้เคียงไหม
        st.write("รายชื่อโฟลเดอร์ทั้งหมด:", os.listdir("."))

# Layout
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown("### 📋 ข้อมูลผู้ป่วย")
    user_input = st.text_area("ระบุอาการ:", placeholder="ระบุอาการทางคลินิกที่นี่...", height=150)
    analyze_btn = st.button("🔍 วิเคราะห์อาการ")

with col_right:
    if analyze_btn and user_input:
        with st.spinner("🔄 กำลังประมวลผล..."):
            try:
                response = st.session_state.agent.query(user_input)
                
                st.markdown("### 📊 รายงานผลการวิเคราะห์")
                # Use a container for the report with a border
                with st.container(border=True):
                    st.markdown(response) # Pure markdown for correct table rendering
                
                with st.expander("🛠️ ระบบการทำงาน (Log Trace)"):
                    if os.path.exists("logs/agent.log"):
                        with open("logs/agent.log", "r", encoding='utf-8') as f:
                            st.code("".join(f.readlines()[-20:]), language="text")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("💡 กรุณาระบุอาการด้านซ้ายเพื่อเริ่มการวิเคราะห์")
