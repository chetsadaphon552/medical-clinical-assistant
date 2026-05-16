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

# 🏥 ระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical Decision Support System - CDSS)

**ระบบ Agentic RAG อัจฉริยะสำหรับวิเคราะห์อาการและวินิจฉัยโรคเบื้องต้นระดับ Medical Grade**

---

## 🌟 การอัปเดตล่าสุด (Latest Updates)

- 🩺 **Professional Clinical UI**: ปรับปรุงหน้าจอ (UI) ใหม่ทั้งหมด ถอดอีโมจิและใช้โทนสี Medical Blue/Slate ให้ดูเป็นทางการระดับซอฟต์แวร์โรงพยาบาล (HIS)
- 🛡️ **Strict Input Guardrails**: มีระบบคัดกรองคำถามที่ทรงพลัง หากผู้ป่วยพิมพ์คำทักทาย ลาก่อน มุกตลก หรือคำถามที่ไม่มีอาการเจ็บป่วย AI จะปฏิเสธการวิเคราะห์ทันที
- 🧠 **Critical Evaluation (Anti-Hallucination)**: AI มีระบบ "นักสืบ" ที่คอยโต้แย้งข้อมูลจาก RAG หากฐานข้อมูลดึงโรคที่ไม่สอดคล้องกับอาการของผู้ป่วยขึ้นมา AI จะระบุข้อควรระวังและลดความน่าเชื่อถือของโรคนั้นลง
- 🇹🇭 **100% Thai Medical Dictionary**: บังคับแปลชื่อโรคและศัพท์ทางการแพทย์เป็นภาษาไทยเป๊ะๆ ตามพจนานุกรม 22 โรคที่ฝังอยู่ในระบบ ป้องกันการแปลทับศัพท์แปลกๆ (เช่น Dengue -> ไข้เลือดออก)
- 📊 **Strict Top-3 Formatting**: บังคับให้ AI แสดงผลเฉพาะ **3 อันดับโรคที่เป็นไปได้สูงสุด** ในรูปแบบ Markdown Table ที่สะอาดตาและอ่านง่าย

## 📚 ฐานข้อมูลโรค (Supported Conditions)
ระบบนี้รองรับการวิเคราะห์อาการจากฐานข้อมูลทั้งหมด **22 โรค** ครอบคลุมตั้งแต่หวัดทั่วไป ภูมิแพ้ ปอดอักเสบ ไข้เลือดออก ไปจนถึงเบาหวาน
👉 [ดูรายชื่อโรคทั้งหมดที่นี่ (DISEASE_LIST.md)](./DISEASE_LIST.md)

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

- **LLM**: Qwen2.5-Omni-7B (ผ่าน DashScope API)
- **Framework**: LangChain 
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: BAAI/bge-base-en-v1.5
- **UI**: Streamlit
- **Deployment**: Docker Compose & Hugging Face Spaces

---

## 🚀 การติดตั้งและใช้งาน (Installation & Usage)

### 1. ใช้งานผ่าน Cloud (Hugging Face)
สามารถเข้าใช้งานได้ทันทีที่: `https://huggingface.co/spaces/chetsadaphon66/medical-clinical-assistant`

### 2. ใช้งานภายในเครื่อง (Local Run - แนะนำ)
คุณสามารถรันระบบบนเครื่องตัวเองได้ 2 วิธี:

**วิธีที่ 2.1: รันด้วย Python / Streamlit โดยตรง (เหมาะสำหรับ Dev)**
```bash
# 1. ติดตั้งไลบรารี
pip install -r requirements.txt

# 2. รันหน้าเว็บ
python -m streamlit run app.py
```

**วิธีที่ 2.2: รันด้วย Docker Compose (เหมาะสำหรับ Production)**
```bash
# สร้างและรัน Container
docker-compose up --build -d
```

## 🔐 การตั้งค่าสภาพแวดล้อม (Environment Variables)
สร้างไฟล์ `.env` ในโฟลเดอร์หลัก และใส่ข้อมูลดังนี้:
```env
# Qwen API Settings
QWEN_API_KEY=your_api_key_here
QWEN_MODEL=qwen2.5-omni-7b
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Vector DB (บังคับใช้ Path นี้)
VECTOR_DB_PATH=data/vectordb/vector_store
```

---

⚠️ **คำเตือน (Disclaimer)**: ระบบนี้ถูกออกแบบมาเพื่อเป็นเครื่องมือสนับสนุนการตัดสินใจเบื้องต้นเท่านั้น การวินิจฉัยขั้นสุดท้ายและการวางแผนการรักษาต้องขึ้นอยู่กับดุลยพินิจของแพทย์ผู้เชี่ยวชาญเสมอ
