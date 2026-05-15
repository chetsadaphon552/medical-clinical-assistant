import streamlit as st
import os
import sys
import logging
from io import StringIO
from src.agent import MedicalSymptomAssistant

# Page Configuration
st.set_page_config(
    page_title="Clinical CDS Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Medical Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
    .report-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .ddx-item {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .confidence-score {
        float: right;
        color: #28a745;
        font-weight: bold;
    }
    .reasoning-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Helper for capturing logs for Observability
class StreamlitLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.log_stream = StringIO()

    def emit(self, record):
        msg = self.format(record)
        self.log_stream.write(msg + "\n")
        self.placeholder.code(self.log_stream.getvalue())

# Initialize Session State
if 'agent' not in st.session_state:
    with st.spinner("🏥 Initializing Clinical CDS Agent..."):
        st.session_state.agent = MedicalSymptomAssistant()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2773/2773533.png", width=100)
    st.title("Clinical CDS")
    st.markdown("---")
    st.info("**Project:** Medical Symptom Assistant\n\n**Framework:** Agentic RAG\n\n**Model:** Qwen2.5-Omni")
    
    st.warning("""
    ⚠️ **คำเตือน (Disclaimer):**
    ระบบนี้เป็นเครื่องมือช่วยตัดสินใจทางคลินิก (CDSS) เท่านั้น ไม่ใช่การวินิจฉัยโรคขั้นสุดท้าย การตัดสินใจขึ้นอยู่กับดุลยพินิจของแพทย์
    """)
    
    if st.button("🔄 Clear Session"):
        st.rerun()

# Main Header
st.title("🏥 Clinical Decision Support Assistant")
st.subheader("ระบบวิเคราะห์อาการและวินิจฉัยแยกโรคเบื้องต้น")

# User Input Section
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "ระบุอาการแสดงของผู้ป่วย (Patient Presentation):", 
        placeholder="เช่น: ไอและมีน้ำมูกไหลมา 3 วัน ไม่มีไข้...",
        height=100
    )
with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("🔍 วิเคราะห์อาการ")

# Observability Section (Real-time Logs)
with st.expander("🛠️ Agent Reasoning & Observability (ดูการทำงานของระบบ)"):
    log_placeholder = st.empty()
    # Note: In a real app, we would redirect Python logging here
    st.caption("Logs จะแสดงข้อมูลการเรียกใช้ Tool และการประมวลผลของ Agent")

if analyze_button and user_input:
    # Set up logging capture
    log_handler = StreamlitLogHandler(log_placeholder)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(log_handler)
    
    try:
        # Progress indicator
        with st.status("👨‍⚕️ Agent กำลังวิเคราะห์ข้อมูล...", expanded=True) as status:
            st.write("🌐 กำลังแปลภาษาและเตรียมการค้นหา...")
            # Query the agent
            response = st.session_state.agent.query(user_input)
            status.update(label="✅ การวิเคราะห์เสร็จสมบูรณ์", state="complete", expanded=False)

        # Display Results
        st.markdown("### 📊 รายงานการวินิจฉัยแยกโรค (Clinical Report)")
        
        # Display response in a nice format
        st.markdown(f'<div class="report-card">{response}</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    finally:
        # Cleanup log handler
        logging.getLogger().removeHandler(log_handler)

elif analyze_button and not user_input:
    st.warning("กรุณาระบุอาการก่อนทำการวิเคราะห์")

# Footer
st.markdown("---")
st.caption("© 2026 Clinical Decision Support System - Agentic RAG Pipeline")
