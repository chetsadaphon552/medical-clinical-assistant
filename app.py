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

# Advanced Custom Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Report Card Styling */
    .report-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #1f2937;
        margin-top: 20px;
    }
    
    /* DDx Tag Styling */
    .ddx-tag {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9em;
        margin-right: 10px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1e3a8a !important;
    }
    
    /* Button Animation */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 12px;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'agent' not in st.session_state:
    with st.spinner("👨‍⚕️ กำลังเตรียมความพร้อมของระบบ..."):
        try:
            st.session_state.agent = MedicalSymptomAssistant()
        except Exception as e:
            st.error(f"Error initializing system: {e}")

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>🏥 CDS Dashboard</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2773/2773533.png", width=100)
    st.markdown("---")
    st.markdown("🔍 **System Specs**")
    st.write("Model: `Qwen2.5-Omni-7B`")
    st.write("Retriever: `BGE-v1.5` (RAG)")
    st.markdown("---")
    st.markdown("👨‍⚕️ **Clinical Guide**")
    st.caption("ระบบนี้ใช้เพื่อสนับสนุนการวินิจฉัยแยกโรค (DDx) โดยใช้ฐานข้อมูลทางคลินิกอ้างอิง")
    if st.button("🔄 Reset Analysis"):
        if os.path.exists("logs/agent.log"):
            with open("logs/agent.log", "w") as f: f.write("")
        st.rerun()

# Layout
col_main, col_spacer = st.columns([10, 1])

with col_main:
    st.markdown("<h1 class='main-header'>🏥 Clinical Decision Support Assistant</h1>", unsafe_allow_html=True)
    
    # Input Area
    with st.container():
        st.markdown("### 📋 ข้อมูลผู้ป่วย (Patient Presentation)")
        user_input = st.text_area("", placeholder="ระบุอาการแสดงทางคลินิก (เช่น: ไอและมีน้ำมูกไหล ไม่มีไข้...)", height=120, label_visibility="collapsed")
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            analyze_btn = st.button("🚀 เริ่มการวิเคราะห์เชิงลึก (Run Analysis)", use_container_width=True)

    # Execution Area
    if analyze_btn and user_input:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
            if percent_complete == 20: status_text.text("🌐 กำลังแปลภาษา (Multilingual Translation)...")
            if percent_complete == 50: status_text.text("🔍 กำลังค้นหาฐานข้อมูล (RAG Semantic Search)...")
            if percent_complete == 80: status_text.text("🧠 กำลังวิเคราะห์ผลลัพธ์ (Clinical Reasoning)...")
            
        try:
            response = st.session_state.agent.query(user_input)
            status_text.empty()
            progress_bar.empty()
            
            # Display Result
            st.markdown("### 📊 รายงานผลการวิเคราะห์ทางคลินิก (Clinical Report)")
            st.markdown(f"""
            <div class="report-card">
                {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Observability Trace
            with st.expander("🛠️ Advanced Agent Trace (Observability)"):
                if os.path.exists("logs/agent.log"):
                    with open("logs/agent.log", "r", encoding='utf-8') as f:
                        logs = f.readlines()
                        st.code("".join(logs[-30:]), language="text")
                else:
                    st.info("No logs available")
                    
        except Exception as e:
            st.error(f"❌ Analysis Failed: {e}")
            
    elif analyze_btn and not user_input:
        st.warning("⚠️ กรุณาระบุอาการผู้ป่วยก่อนทำการวิเคราะห์")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280;'>© 2026 Medical AI Support System | Empowering Physicians with Agentic RAG</p>", unsafe_allow_html=True)
