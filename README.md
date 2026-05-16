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

## 📂 โครงสร้างโปรเจกต์ (Project Structure)
```text
medical-symptom-assistant/
├── app.py                  # หน้าเว็บ UI หลัก (Streamlit)
├── api.py                  # ระบบ REST API (FastAPI) สำหรับเรียกผ่าน cURL/Postman
├── compare_models.py       # สคริปต์ A/B Testing สำหรับเปรียบเทียบโมเดล Embedding
├── DISEASE_LIST.md         # พจนานุกรมรายชื่อโรค 22 โรคที่รองรับ
├── docker-compose.yml      # ไฟล์ตั้งค่า Docker Compose สำหรับรันบน Server
├── Dockerfile              # ไฟล์สำหรับสร้าง Docker Image
├── requirements.txt        # รายการไลบรารีที่ต้องใช้
├── .env                    # (คุณต้องสร้างเอง) ไฟล์เก็บรหัส QWEN_API_KEY
├── data/
│   ├── raw/                # ข้อมูลดิบ (Dataset)
│   ├── processed/          # ข้อมูลที่ผ่านการจัดเตรียม (documents.json)
│   └── vectordb/           # ฐานข้อมูล Vector (FAISS Index)
└── src/
    ├── agent.py            # หัวสมองหลัก (LLM Logic, Translation, System Prompts)
    ├── tools.py            # เครื่องมือของ AI (Vector Search, FAISS Retrieval)
    └── setup_vectordb.py   # สคริปต์หั่น Chunk และสร้างฐานข้อมูลเวกเตอร์
```

## 🚀 การติดตั้งและใช้งาน (Installation & Usage)

### 1. ใช้งานผ่าน Web UI (Streamlit)
**รันบนเครื่อง Local:**
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
**รันด้วย Docker:**
```bash
docker-compose up --build -d
```

### 2. ใช้งานผ่าน REST API (FastAPI)
หากคุณต้องการนำระบบนี้ไปเชื่อมต่อกับ Frontend อื่น หรือ LINE Chatbot คุณสามารถเปิดรันเซิร์ฟเวอร์ API ได้ด้วยคำสั่ง:
```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

**ตัวอย่างการยิง cURL Command เพื่อขอรับการวิเคราะห์อาการ:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "symptoms": "ปวดศีรษะมาก มีไข้สูง และปวดกระบอกตามา 2 วัน"
}'
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
