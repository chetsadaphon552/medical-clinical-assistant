---
title: Medical Clinical Assistant
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# 🏥 ระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical Decision Support System — CDSS)

**ระบบ Agentic RAG สำหรับวิเคราะห์อาการและวินิจฉัยโรคเบื้องต้น พัฒนาด้วย Typhoon LLM และ FAISS Vector Database**

---

## ✨ ความสามารถหลัก

- 🔍 **Differential Diagnosis** — วิเคราะห์อาการและแสดง Top 3 โรคที่เป็นไปได้ พร้อม Confidence Score
- � **Condition Lookup** — ดูรายละเอียดอาการของโรคเฉพาะจากฐานข้อมูล
- 🧠 **Critical Evaluation** — AI โต้แย้งข้อมูลจาก RAG หากโรคที่ดึงมาไม่สอดคล้องกับอาการจริง
- 🛡️ **Input Guardrails** — ปฏิเสธคำถามที่ไม่เกี่ยวกับการแพทย์ และโรคนอกฐานข้อมูล 22 โรค
- 💬 **Multi-turn Chat** — จำบริบทการสนทนาได้ สามารถถามต่อเนื่องได้
- 🇹🇭 **ภาษาไทย 100%** — รองรับทั้งภาษาไทยและอังกฤษ ตอบกลับเป็นภาษาไทยเสมอ

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Typhoon v2.5 30B (`typhoon-v2.5-30b-a3b-instruct`) by SCB 10X |
| Agent Framework | LangChain |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Embedding Model | `BAAI/bge-base-en-v1.5` (768 dims, Cosine Similarity) |
| UI | Streamlit |
| Deployment | Docker Compose & Hugging Face Spaces |

---

## 🏗️ System Architecture

```
User Input (Thai/English)
        │
        ▼
┌───────────────────┐
│   Translation     │  Thai → English (for routing only)
│   (Typhoon LLM)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Tool Router     │  Intent classification
│   (Typhoon LLM)   │
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
Tool 1     Tool 2
search_    get_condition
symptoms   _details
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│  FAISS Vector   │
│  Database       │
│  486 chunks     │
│  22 diseases    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Response        │
│ Generation      │
│ (Typhoon LLM)   │
└─────────────────┘
```

---

## 🔧 Agent Tools

### Tool 1: `search_symptoms`
ค้นหาโรคที่เป็นไปได้จากอาการที่ผู้ป่วยรายงาน
```
Input : "มีไข้สูง ปวดหัว ผื่นขึ้น"
Output: Top 3 โรค + Confidence Score + Clinical Analysis
```

### Tool 2: `get_condition_details`
ดึงรายละเอียดทางคลินิกของโรคเฉพาะ
```
Input : "ไข้เลือดออก"
Output: คำจำกัดความ + อาการแสดงหลัก
```

> 📄 ดูรายละเอียด Tools เพิ่มเติมได้ที่ [TOOLS.md](./TOOLS.md)

---

## 📚 ฐานข้อมูลโรค

รองรับ **22 โรค** ครอบคลุมตั้งแต่หวัดทั่วไป ภูมิแพ้ ปอดอักเสบ ไข้เลือดออก ไปจนถึงเบาหวาน

👉 [ดูรายชื่อโรคทั้งหมด (DISEASE_LIST.md)](./DISEASE_LIST.md)

---

## 📂 โครงสร้างโปรเจกต์

```
medical-symptom-assistant/
├── app.py                  # Web UI (Streamlit) — Chat interface
├── api.py                  # REST API (FastAPI)
├── DISEASE_LIST.md         # รายชื่อ 22 โรคที่รองรับ
├── TOOLS.md                # เอกสาร Agent Tools
├── docker-compose.yml      # Docker Compose config
├── Dockerfile              # Docker image
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # documents.json
│   └── vectordb/           # FAISS index + chunks
└── src/
    ├── agent.py            # CDSS Agent (routing, reasoning, response generation)
    ├── tools.py            # RAG Tools (search_symptoms, get_condition_details)
    └── setup_vectordb.py   # Vector store builder
```

---

## 🚀 การติดตั้งและใช้งาน

### Local (Streamlit)
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### Docker
```bash
docker-compose up --build -d
```

---

## 🔐 Environment Variables

สร้างไฟล์ `.env` และใส่ค่าดังนี้:

```env
# Typhoon API (https://opentyphoon.ai)
TYPHOON_API_KEY=your_api_key_here
TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct
TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Vector DB
VECTOR_DB_PATH=data/vectordb/vector_store
```

---

> ⚠️ **Disclaimer**: ระบบนี้เป็นเครื่องมือสนับสนุนการตัดสินใจเบื้องต้นเท่านั้น การวินิจฉัยขั้นสุดท้ายต้องขึ้นอยู่กับดุลยพินิจของแพทย์ผู้เชี่ยวชาญเสมอ
