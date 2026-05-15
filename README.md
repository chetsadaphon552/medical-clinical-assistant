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

# 👨‍⚕️ ระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical Decision Support System - CDSS)

**ระบบ Agentic RAG อัจฉริยะสำหรับบุคลากรทางการแพทย์ เพื่อการวินิจฉัยแยกโรค (Differential Diagnosis) ที่แม่นยำและรวดเร็ว**

---

## 🌟 ฟีเจอร์เด่น (Key Features)

- 🤖 **Agentic RAG Pipeline**: ใช้กระบวนการคิดของ AI Agent ในการดึงข้อมูลจากฐานข้อมูลอาการจริง (Clinical Evidence)
- 📊 **Strict Clinical Reporting**: รายงานผลในรูปแบบตาราง Markdown ที่สวยงาม แยกคอลัมน์ลำดับโรคและคะแนนความมั่นใจ (Confidence Score) ชัดเจน
- 🧠 **100% Thai Clinical Analysis**: ระบบวิเคราะห์อาการเป็นภาษาไทยระดับทางการ (Professional Thai) โดยแปลหลักฐานจากภาษาอังกฤษให้อัตโนมัติ
- 🔍 **Semantic Search Optimization**: ใช้ Vector Database (FAISS) ร่วมกับ Multilingual Embeddings เพื่อการสืบค้นที่แม่นยำที่สุด
- 🚀 **Full CI/CD Integration**: ระบบทดสอบและ Deploy อัตโนมัติจาก GitHub ไปยัง Hugging Face Spaces

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

- **LLM**: Qwen2.5-Omni-7B (ผ่าน DashScope API)
- **Framework**: LangChain & LangGraph
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **UI**: Streamlit (Professional Clinical Dashboard)
- **CI/CD**: GitHub Actions & Hugging Face Hub SDK

## 📋 ตัวอย่างการแสดงผล (Output Example)

เมื่อผู้ป่วยระบุอาการ เช่น "หนาวสั่นและไอต่อเนื่อง" ระบบจะแสดงผลดังนี้:

| ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
| :--- | :--- | :--- |
| 1 | ปอดอักเสบ (Pneumonia) | 0.85 |
| 2 | หวัดทั่วไป (Common Cold) | 0.78 |
| 3 | ไข้หวัดใหญ่ (Influenza) | 0.75 |

---

## 🚀 การติดตั้งและใช้งาน (Installation)

### 1. ใช้งานผ่าน Cloud (Hugging Face)
สามารถเข้าใช้งานได้ทันทีที่: `https://huggingface.co/spaces/chetsadaphon66/medical-clinical-assistant`

### 2. ใช้งานภายในเครื่อง (Local via Docker)
```bash
# Clone โปรเจกต์
git clone https://github.com/chetsadaphon552/medical-clinical-assistant.git
cd medical-clinical-assistant

# รันด้วย Docker Compose
docker-compose up --build web
```

## 🔐 การตั้งค่าสภาพแวดล้อม (Environment Variables)
สร้างไฟล์ `.env` และเพิ่มคีย์ดังนี้:
```env
QWEN_API_KEY=your_api_key_here
QWEN_MODEL=qwen2.5-7b-instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

⚠️ **คำเตือน (Disclaimer)**: ระบบนี้ถูกออกแบบมาเพื่อเป็นเครื่องมือสนับสนุนการตัดสินใจสำหรับบุคลากรทางการแพทย์เท่านั้น การวินิจฉัยขั้นสุดท้ายและการวางแผนการรักษาต้องขึ้นอยู่กับดุลยพินิจของแพทย์ผู้เชี่ยวชาญเสมอ
