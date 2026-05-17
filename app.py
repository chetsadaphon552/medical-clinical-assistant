import streamlit as st
import os
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
    .stApp { background-color: #F4F7F6; }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #2D3748 !important;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        border: 1px solid #CBD5E0;
    }
    th {
        background-color: #E2E8F0;
        color: #2D3748;
        font-weight: bold;
        text-align: left;
        padding: 10px 12px;
        border: 1px solid #CBD5E0;
    }
    td {
        padding: 10px 12px;
        border: 1px solid #CBD5E0;
    }

    section[data-testid="stSidebar"] { background-color: #1A365D !important; }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* Chat bubbles */
    .user-bubble {
        background-color: #1A365D;
        color: white !important;
        padding: 10px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E0;
        padding: 14px 18px;
        border-radius: 16px 16px 16px 4px;
        margin: 8px 0;
        max-width: 95%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent" not in st.session_state:
    with st.spinner("Initializing System..."):
        try:
            st.session_state.agent = MedicalSymptomAssistant()
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>CDSS Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Model:** `{st.session_state.agent.llm.model_name}`")
    st.markdown(f"**History:** {len(st.session_state.agent.chat_history)} turns")
    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.clear_history()
        if os.path.exists("logs/agent.log"):
            with open("logs/agent.log", "w") as f:
                f.write("")
        st.rerun()

    st.markdown("---")
    with st.expander("📋 System Trace Logs"):
        if os.path.exists("logs/agent.log"):
            with open("logs/agent.log", "r", encoding="utf-8") as f:
                lines = f.readlines()[-30:]
            st.code("".join(lines), language="text")

# Header
st.title("⚕️ Clinical Decision Support System")
st.markdown("ระบบสนับสนุนการตัดสินใจทางคลินิก (CDSS)")
st.markdown("---")

# Chat History Display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            with st.container(border=True):
                st.markdown(msg["content"])

# Input Area
st.markdown("---")
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Chief Complaint / Symptoms:",
        placeholder="ระบุอาการแสดงทางคลินิกของผู้ป่วย...",
        label_visibility="collapsed",
        key="user_input"
    )
with col_btn:
    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

# Process Query
if analyze_btn and user_input.strip():
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Processing Clinical Data..."):
        try:
            response = st.session_state.agent.query(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            })

    st.rerun()
