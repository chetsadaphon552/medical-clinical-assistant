# 🔧 Agent Tools — Clinical Decision Support System

ระบบ CDSS ใช้ **Agentic RAG** โดย Agent จะเลือกใช้ Tool ที่เหมาะสมโดยอัตโนมัติตาม intent ของผู้ใช้

---

## 🏗️ Architecture Overview

```
User Input (Thai/English)
        │
        ▼
┌───────────────────┐
│   Translation     │  Thai → English (for routing only)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Tool Router     │  LLM-based intent classification
│   (Typhoon LLM)   │
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
Tool 1     Tool 2
```

---

## 🛠️ Tool 1: `search_symptoms`

### วัตถุประสงค์
ค้นหาโรคที่เป็นไปได้จากอาการที่ผู้ป่วยรายงาน (**Differential Diagnosis**)

### เรียกใช้เมื่อ
- ผู้ใช้อธิบายอาการทางกายภาพที่กำลังเป็นอยู่
- ไม่มีการระบุชื่อโรคเฉพาะ

### Input
| Parameter | Type | Description |
|-----------|------|-------------|
| `symptoms` | `str` | คำอธิบายอาการ เช่น `"fever, cough, sore throat"` |
| `k` | `int` | จำนวนผลลัพธ์ที่ต้องการ (default: 5) |

### กระบวนการทำงาน
```
1. Query Expansion
   "fever cough" → "patient experiencing fever cough symptoms medical condition"

2. Semantic Embedding
   SentenceTransformer (BAAI/bge-base-en-v1.5) → vector [768 dims]

3. FAISS Search
   Cosine similarity search ใน 486 chunks

4. Relevance Filtering
   threshold = 0.60 → กรองผลลัพธ์ที่ไม่เกี่ยวข้องออก

5. Deduplication
   เลือก 1 chunk ต่อ 1 โรค (unique conditions only)
```

### Output Format
```
Condition: dengue
Confidence Score: 0.82
Clinical Content: [patient symptom descriptions...]
Source: gretel
```

### ตัวอย่างคำถามที่ trigger tool นี้
```
✅ "มีไข้สูง หนาวสั่น ปวดหัว ปวดกล้ามเนื้อ"
✅ "ปวดท้องน้อย ปัสสาวะแสบขัด ปัสสาวะบ่อย"
✅ "I have high fever, joint pain and rash"
✅ "ลูกมีผื่นขึ้น คัน มีตุ่มน้ำใส"
```

---

## 🛠️ Tool 2: `get_condition_details`

### วัตถุประสงค์
ดึงข้อมูลรายละเอียดของโรคเฉพาะจากฐานข้อมูล (**Condition Lookup**)

### เรียกใช้เมื่อ
- ผู้ใช้ถามเกี่ยวกับโรคที่ระบุชื่อชัดเจน
- ต้องการทราบอาการ คำจำกัดความ หรือข้อมูลทางคลินิกของโรคนั้น

### Input
| Parameter | Type | Description |
|-----------|------|-------------|
| `condition_name` | `str` | ชื่อโรค เช่น `"diabetes"`, `"dengue"` |

### กระบวนการทำงาน
```
1. Enhanced Query
   "diabetes" → "diabetes disease symptoms treatment clinical information"

2. Semantic Embedding + FAISS Search
   ค้นหา 30 candidates

3. Fuzzy Matching
   จับคู่ชื่อโรค (exact / partial match)

4. Fallback
   ถ้าไม่พบ exact match → ใช้ semantic similarity > 0.5

5. Aggregate
   รวมทุก chunks ของโรคนั้น → ข้อมูลครบถ้วน
```

### Output Format
```
=== รายละเอียดโรค: diabetes ===

[patient symptom descriptions from all chunks...]

--- Metadata ---
Condition: diabetes
Category: General
Severity: moderate
Retrieved Chunks: 1
```

### ตัวอย่างคำถามที่ trigger tool นี้
```
✅ "อาการของโรคเบาหวาน"
✅ "ไข้เลือดออกเป็นยังไง"
✅ "ภูมิแพ้อาการเป็นยังไง"
✅ "tell me about pneumonia"
✅ "โรคตับอักเสบ เอ มีอาการอะไรบ้าง"
```

---

## 🚦 Tool Routing Logic

```
User Query
    │
    ├─► [Python Guard 1] มีชื่อโรคภาษาไทย/alias?
    │       YES → get_condition_details
    │
    ├─► [LLM Router] Typhoon วิเคราะห์ intent
    │       disease name + asking about it → get_condition_details
    │       physical symptoms described   → search_symptoms
    │       non-medical / vague           → none (ปฏิเสธ)
    │
    └─► [Python Guard 2] search_symptoms: มี symptom keyword?
            NO → ปฏิเสธ (เช่น "อาการโรคผอม")
            YES → ดำเนินการต่อ
```

---

## 🛡️ Safety Guards

| Guard | ประเภท | วัตถุประสงค์ |
|-------|--------|-------------|
| Whitelist (22 โรค) | Python | บล็อกโรคนอกฐานข้อมูลก่อนเรียก LLM |
| Symptom Keywords | Python | ป้องกัน vague queries เข้า search_symptoms |
| Disease Aliases | Python | จับ informal names เช่น "ภูมิ", "หวัด", "ปอด" |
| Confidence Threshold (0.60) | FAISS | กรองผลลัพธ์ที่ไม่เกี่ยวข้อง |
| Post-filter | Python | ลบโรคนอก whitelist ออกจาก RAG results |

---

## 📊 Vector Store

| Property | Value |
|----------|-------|
| Embedding Model | `BAAI/bge-base-en-v1.5` |
| Dimension | 768 |
| Index Type | FAISS IndexFlatIP (cosine similarity) |
| Total Chunks | 486 |
| Documents | 22 โรค |
| Avg Chunks/Doc | 22.1 |
| Similarity Metric | Cosine similarity (normalized L2) |

---

## 🤖 LLM

**Typhoon v2.5 30B** — AI ภาษาไทยโดย SCB 10X ใช้ทำ 3 งานในระบบนี้:

| งาน | คำอธิบาย |
|-----|---------|
| แปลภาษา | แปลคำถามไทย → อังกฤษ เพื่อค้นหาในฐานข้อมูล |
| เลือก Tool | วิเคราะห์ว่าผู้ใช้ถามอะไร แล้วเลือก Tool ที่เหมาะสม |
| สร้างคำตอบ | นำข้อมูลจาก RAG มาเขียนเป็นรายงานภาษาไทย |

> เรียกใช้ผ่าน OpenAI-compatible API — ใช้งานเหมือน ChatGPT แต่ชี้ endpoint ไปที่ `api.opentyphoon.ai` แทน
