import json
import logging
import os
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.tools import search_symptoms, get_condition_details

load_dotenv()

# ---------------------------------------------------------------------------
# Config — all values come from .env
# ---------------------------------------------------------------------------
API_KEY   = os.getenv("TYPHOON_API_KEY")
MODEL     = os.getenv("TYPHOON_MODEL",    "typhoon-v2.5-30b-a3b-instruct")
BASE_URL  = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cdss.agent")

# ---------------------------------------------------------------------------
# Whitelist — 22 conditions that exist in the vector store.
# Queries outside this set are rejected before any LLM call.
# ---------------------------------------------------------------------------
SUPPORTED_CONDITIONS: set[str] = {
    "cervical spondylosis", "impetigo", "urinary tract infection", "arthritis",
    "dengue", "common cold", "drug reaction", "fungal infection", "malaria",
    "allergy", "bronchial asthma", "varicose veins", "migraine", "hypertension",
    "gastroesophageal reflux disease", "pneumonia", "psoriasis", "diabetes",
    "jaundice", "chicken pox", "typhoid", "hepatitis a",
}

DISEASE_TH: dict[str, str] = {
    "cervical spondylosis": "โรคกระดูกคอเสื่อม",
    "impetigo": "โรคพุพอง",
    "urinary tract infection": "โรคติดเชื้อทางเดินปัสสาวะ",
    "arthritis": "โรคข้ออักเสบ",
    "dengue": "ไข้เลือดออก",
    "common cold": "หวัดทั่วไป",
    "drug reaction": "การแพ้ยา",
    "fungal infection": "การติดเชื้อเชื้อรา",
    "malaria": "มาลาเรีย",
    "allergy": "ภูมิแพ้",
    "bronchial asthma": "โรคหอบหืด",
    "varicose veins": "เส้นเลือดขอด",
    "migraine": "ไมเกรน",
    "hypertension": "โรคความดันโลหิตสูง",
    "gastroesophageal reflux disease": "โรคกรดไหลย้อน",
    "pneumonia": "ปอดอักเสบ",
    "psoriasis": "โรคสะเก็ดเงิน",
    "diabetes": "โรคเบาหวาน",
    "jaundice": "ดีซ่าน",
    "chicken pox": "โรคอีสุกอีใส",
    "typhoid": "ไข้ไทฟอยด์",
    "hepatitis a": "โรคตับอักเสบ เอ",
}

NOT_FOUND_MSG = """❌ ไม่พบข้อมูลของโรคนี้ในฐานข้อมูล

ระบบรองรับเฉพาะ **22 โรค** ต่อไปนี้เท่านั้น:

| # | ภาษาไทย | English |
|---|---------|---------|
| 1 | โรคกระดูกคอเสื่อม | Cervical Spondylosis |
| 2 | โรคพุพอง | Impetigo |
| 3 | โรคติดเชื้อทางเดินปัสสาวะ | Urinary Tract Infection |
| 4 | โรคข้ออักเสบ | Arthritis |
| 5 | ไข้เลือดออก | Dengue |
| 6 | หวัดทั่วไป | Common Cold |
| 7 | การแพ้ยา | Drug Reaction |
| 8 | การติดเชื้อเชื้อรา | Fungal Infection |
| 9 | มาลาเรีย | Malaria |
| 10 | ภูมิแพ้ | Allergy |
| 11 | โรคหอบหืด | Bronchial Asthma |
| 12 | เส้นเลือดขอด | Varicose Veins |
| 13 | ไมเกรน | Migraine |
| 14 | โรคความดันโลหิตสูง | Hypertension |
| 15 | โรคกรดไหลย้อน | Gastroesophageal Reflux Disease |
| 16 | ปอดอักเสบ | Pneumonia |
| 17 | โรคสะเก็ดเงิน | Psoriasis |
| 18 | โรคเบาหวาน | Diabetes |
| 19 | ดีซ่าน | Jaundice |
| 20 | โรคอีสุกอีใส | Chicken Pox |
| 21 | ไข้ไทฟอยด์ | Typhoid |
| 22 | โรคตับอักเสบ เอ | Hepatitis A |

💡 กรุณาเลือกโรคจากรายการข้างต้น หรืออธิบายอาการของผู้ป่วยเพื่อให้ระบบช่วยวิเคราะห์"""

# ---------------------------------------------------------------------------
# Prompts — defined as module-level constants for easy maintenance
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Clinical Decision Support Assistant (CDSS) — an intelligent clinical symptom analysis system.

## Output Language
- ALWAYS respond in Thai (ภาษาไทย) only.
- English is allowed ONLY inside parentheses for technical terms, e.g. ไข้สูง (High fever).
- FORBIDDEN characters: Chinese (典型, 週, 糖尿病), Japanese, Korean, Vietnamese, or any non-Thai/non-English script.
- If a term cannot be translated accurately, keep the English term in parentheses rather than guessing.

## Clinical Reasoning
- Base ALL analysis strictly on the provided RAG context. Never fabricate data.
- Apply differential diagnosis: compare and contrast conditions to justify rankings.
- Rank conditions by confidence score (descending).
- Apply critical evaluation: if a RAG-retrieved condition does not match the reported symptoms, explicitly state the mismatch.

## Medical Terminology (Thai translations)
Fatigue→อ่อนเพลีย | Weakness→อ่อนแรง | Chills→หนาวสั่น | Fever→ไข้ | Headache→ปวดหัว
Dizziness→เวียนหัว | Nausea→คลื่นไส้ | Vomiting→อาเจียน | Sweating→เหงื่อออก
Shortness of breath→หายใจลำบาก | Chest tightness→แน่นหน้าอก | Chest pain→ปวดหน้าอก
Cough→ไอ | Sputum→เสมหะ | Wheezing→หายใจมีเสียงหวีด | Sore throat→เจ็บคอ
Runny nose→น้ำมูกไหล | Nasal congestion→คัดจมูก | Abdominal pain→ปวดท้อง
Diarrhea→ท้องเสีย | Constipation→ท้องผูก | Loss of appetite→เบื่ออาหาร
Frequent urination→ปัสสาวะบ่อย | Excessive thirst→กระหายน้ำมาก | Numbness→ชา
Blurred vision→มองเห็นไม่ชัด | Weight loss→น้ำหนักลด | Rash→ผื่น | Itching→คัน
Swelling→บวม | Red spots→จุดแดง | Joint pain→ปวดข้อ | Muscle pain→ปวดกล้ามเนื้อ
Back pain→ปวดหลัง | Neck pain→ปวดคอ | Stiffness→ตึง/แข็ง
Pain behind eyes→ปวดหลังลูกตา | Sensitivity to light→ไวต่อแสง (Photophobia)
Sensitivity to sound→ไวต่อเสียง (Phonophobia) | Glucose→กลูโคส/น้ำตาลในเลือด
Insulin→อินซูลิน | Weekly→รายสัปดาห์
"""

SEARCH_SYMPTOMS_PROMPT = """\
## Task
Perform differential diagnosis based on the patient's reported symptoms and the RAG context below.

## Patient Symptoms
{symptoms}

## RAG Context
{rag}

## Output Format (strict)
### รายชื่อโรคที่เป็นไปได้ (Possible Conditions)
Present ONLY conditions that appear in the RAG context above with a real confidence score.
- Show maximum 3 conditions, minimum 1.
- If RAG contains only 2 conditions, show only 2 rows — do NOT invent a 3rd row.
- Sort by confidence score descending (row 1 > row 2 > row 3).
- Use this exact table format — Thai name first, English in parentheses:

| ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
|-------|-----------|-----------------|
| 1 | ชื่อภาษาไทย (English) | 0.XX |
| 2 | ชื่อภาษาไทย (English) | 0.XX |

Disease name mapping (use exactly):
Dengue→ไข้เลือดออก | Typhoid→ไข้ไทฟอยด์ | Pneumonia→ปอดอักเสบ | Diabetes→โรคเบาหวาน
Migraine→ไมเกรน | Malaria→มาลาเรีย | Allergy→ภูมิแพ้ | Common Cold→หวัดทั่วไป
Arthritis→โรคข้ออักเสบ | Hypertension→โรคความดันโลหิตสูง | Jaundice→ดีซ่าน
Chicken Pox→โรคอีสุกอีใส | Hepatitis A→โรคตับอักเสบ เอ | Psoriasis→โรคสะเก็ดเงิน
Impetigo→โรคพุพอง | Bronchial Asthma→โรคหอบหืด | Fungal Infection→การติดเชื้อเชื้อรา
Drug Reaction→การแพ้ยา | Urinary Tract Infection→โรคติดเชื้อทางเดินปัสสาวะ
Varicose Veins→เส้นเลือดขอด | Cervical Spondylosis→โรคกระดูกคอเสื่อม
Gastroesophageal Reflux Disease→โรคกรดไหลย้อน

### บทวิเคราะห์ทางคลินิก (Clinical Analysis)
- Explain why each condition matches or does not match the reported symptoms.
- Explicitly flag any RAG condition whose key symptoms are absent from the patient's report (Critical Evaluation).

### ข้อพิจารณาเพิ่มเติม (Clinical Considerations)
- Suggest next diagnostic steps (lab tests, imaging, history questions).
- Note red-flag symptoms that require urgent attention if applicable.
"""

CONDITION_DETAILS_PROMPT = """\
## Task
Summarize the clinical profile of the requested condition using ONLY the RAG context below.
Do NOT fabricate any information not present in the RAG context.

## User Request
{query}

## RAG Context
{rag}

## Output Format
### คำจำกัดความ (Definition)
2–3 sentences describing the condition.

### อาการแสดงหลัก (Main Symptoms)
Numbered list of the most frequently reported symptoms (5–10 items), sorted by frequency.
Format: 1. ชื่ออาการภาษาไทย (English term)

Rules:
- Each symptom must have a unique Thai translation — never repeat the same Thai word for different symptoms.
- If a term cannot be translated accurately, use the English term in parentheses.
- Do NOT add sections (treatment, considerations) if the RAG context does not contain that information.
"""

TOOL_ROUTER_PROMPT = """\
You are a medical query router. Classify the user query into exactly one of three tools.

## Tool Definitions
1. get_condition_details — user asks about a NAMED disease (symptoms, definition, info, details)
   Trigger phrases: อาการของ, อาการโรค, รายละเอียด, บอกเกี่ยวกับ, เป็นยังไง, symptoms of, tell me about, what is
2. search_symptoms — user describes SPECIFIC PHYSICAL symptoms they/patient are experiencing (no disease name)
   Valid symptoms: fever, pain, cough, rash, nausea, vomiting, fatigue, swelling, itching, dizziness, etc.
   Trigger phrases: มีไข้, ปวด, คัน, เหนื่อย, ไอ, บวม, I have, I feel, patient has
3. none — ANY of the following:
   - Greeting, farewell, joke, or non-medical topic
   - Vague complaints that are NOT specific physical symptoms (e.g. "ไม่สบาย", "เหนื่อยๆ", "ผอม", "อ้วน", "อ่อนแอ")
   - Disease names that are NOT in the 22-disease whitelist
   - Lifestyle or nutrition questions

## Decision Rule
- Disease name present + asking about it → get_condition_details
- Specific physical symptoms described, no disease name → search_symptoms
- Vague, non-specific, or non-medical → none

## Examples
"อาการไข้เลือดออก" → {{"tool":"get_condition_details","argument":"dengue"}}
"โรคตับอักเสบมีอาการอะไร" → {{"tool":"get_condition_details","argument":"hepatitis a"}}
"มีไข้ ปวดหัว อ่อนเพลีย" → {{"tool":"search_symptoms","argument":"fever headache fatigue"}}
"ปวดขา บวม เส้นเลือดโป่ง" → {{"tool":"search_symptoms","argument":"leg pain swelling varicose"}}
"อาการโรคผอม" → {{"tool":"none","argument":""}}
"โรคมะเร็ง" → {{"tool":"none","argument":""}}
"เหนื่อยๆ ไม่สบาย" → {{"tool":"none","argument":""}}
"สวัสดี" → {{"tool":"none","argument":""}}

## Query
{query}

Respond with ONLY a JSON object — no explanation, no markdown.
"""


def _is_supported(name: str) -> bool:
    """Return True if name (English) matches any whitelisted condition."""
    n = name.lower().strip()
    if n in SUPPORTED_CONDITIONS:
        return True
    return any(s in n or n in s for s in SUPPORTED_CONDITIONS)


def _filter_rag_by_whitelist(rag_text: str) -> str:
    """Remove blocks for conditions not in the whitelist."""
    lines, skip = [], False
    for line in rag_text.split("\n"):
        m = re.match(r"^Condition:\s*(.+)", line, re.IGNORECASE)
        if m:
            skip = not _is_supported(m.group(1).strip())
            if skip:
                logger.info(f"🚫 Filtered unsupported condition: {m.group(1).strip()}")
        if not skip:
            lines.append(line)
    return "\n".join(lines)


class MedicalSymptomAssistant:
    """Agentic RAG-based Clinical Decision Support System."""

    def __init__(self):
        logger.info("🏥 Initializing CDSS...")
        self.llm = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL,
            temperature=0.1,
            max_tokens=2048,
        )
        self.system_msg = SystemMessage(content=SYSTEM_PROMPT)
        self.tools = {
            "search_symptoms": search_symptoms,
            "get_condition_details": get_condition_details,
        }
        logger.info(f"✅ Model: {MODEL} | Tools: {list(self.tools)}")

    # ------------------------------------------------------------------
    def _route(self, query: str) -> tuple[str, dict]:
        """Use LLM to classify query → (tool_name, tool_input)."""
        try:
            resp = self.llm.invoke(
                [HumanMessage(content=TOOL_ROUTER_PROMPT.format(query=query))]
            )
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                data = json.loads(m.group())
                tool = data.get("tool", "search_symptoms")
                arg  = data.get("argument", query)
                if tool == "get_condition_details":
                    return "get_condition_details", {"condition_name": arg}
                if tool == "none":
                    return "none", {}
            return "search_symptoms", {"symptoms": query}
        except Exception as e:
            logger.error(f"Router error: {e}")
            return "search_symptoms", {"symptoms": query}

    # ------------------------------------------------------------------
    def _translate_to_english(self, text: str) -> str:
        """Translate Thai input to English for better RAG retrieval."""
        resp = self.llm.invoke([
            HumanMessage(
                content=f"Translate to English. Output ONLY the translation, no explanation:\n{text}"
            )
        ])
        return resp.content.strip()

    # ------------------------------------------------------------------
    def query(self, user_input: str) -> str:
        logger.info("=" * 72)
        logger.info(f"USER: {user_input}")
        logger.info("=" * 72)

        original = user_input
        is_thai  = bool(re.search(r"[ก-ฮ]", user_input))

        # 1. Translate Thai → English for RAG
        if is_thai:
            user_input = self._translate_to_english(user_input)
            logger.info(f"TRANSLATED: {user_input}")

        # 2. Route to tool
        tool_name, tool_input = self._route(user_input)
        logger.info(f"ROUTE → {tool_name} | {tool_input}")

        if tool_name == "none":
            return (
                "ขออภัยครับ ระบบนี้ออกแบบมาเพื่อสนับสนุนการตัดสินใจทางคลินิกเท่านั้น "
                "กรุณาระบุอาการของผู้ป่วย หรือชื่อโรคที่ต้องการข้อมูลครับ"
            )

        # 3. Whitelist guard for get_condition_details
        if tool_name == "get_condition_details":
            cond = tool_input.get("condition_name", "")
            if not _is_supported(cond):
                logger.info(f"BLOCKED: '{cond}' not in whitelist")
                return NOT_FOUND_MSG + self._debug_tag(tool_name)

        # 4. Execute tool (RAG retrieval)
        rag_raw = self.tools[tool_name].invoke(tool_input)

        if rag_raw.startswith("❌"):
            return rag_raw + self._debug_tag(tool_name)

        # 5. Post-filter RAG for search_symptoms
        if tool_name == "search_symptoms":
            rag_raw = _filter_rag_by_whitelist(rag_raw)
            if not rag_raw.strip():
                return NOT_FOUND_MSG + self._debug_tag(tool_name)

            scores = [float(s) for s in re.findall(r"Confidence Score:\s*([\d.]+)", rag_raw)]
            if scores and max(scores) < 0.65:
                logger.info(f"LOW CONFIDENCE: max={max(scores):.2f}")
                return (
                    "ขออภัยครับ ไม่พบโรคในฐานข้อมูลที่สอดคล้องกับอาการที่ระบุ "
                    "กรุณาอธิบายอาการทางกายภาพให้ชัดเจนขึ้น เช่น มีไข้ ปวดหัว ไอ ผื่นขึ้น"
                    + self._debug_tag(tool_name)
                )

        # 6. Generate response
        if tool_name == "get_condition_details":
            user_prompt = CONDITION_DETAILS_PROMPT.format(query=original, rag=rag_raw)
        else:
            user_prompt = SEARCH_SYMPTOMS_PROMPT.format(symptoms=original, rag=rag_raw)

        response = self.llm.invoke([self.system_msg, HumanMessage(content=user_prompt)])
        logger.info("✅ Response generated")
        return response.content + self._debug_tag(tool_name)

    # ------------------------------------------------------------------
    @staticmethod
    def _debug_tag(tool_name: str) -> str:
        return f"\n\n---\n*⚙️ Active Tool: **`{tool_name}`***"


# Singleton
assistant = MedicalSymptomAssistant()
