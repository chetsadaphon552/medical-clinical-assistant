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
You are a Clinical Decision Support Assistant (CDSS) built for Thai-language clinical environments.
Your role is to assist clinicians with symptom-based differential diagnosis and condition lookup — not to replace medical judgment.

## Language Rules (non-negotiable)
- Respond EXCLUSIVELY in Thai (ภาษาไทย).
- English is permitted ONLY inside parentheses for medical terms: e.g. ไข้สูง (High fever).
- NEVER output Chinese (典型, 週, 糖尿病, 鎮静劑), Japanese, Korean, Vietnamese, or any non-Thai/non-English characters.
- When a Thai translation is uncertain or awkward, keep the English term in parentheses — do not guess.

## Clinical Reasoning Standards
- Ground every claim in the RAG context provided. Do not fabricate symptoms, diagnoses, or treatments.
- Apply differential diagnosis: for each candidate condition, explain what matches AND what does not match the patient's reported symptoms.
- Rank conditions strictly by confidence score (highest first).
- Flag mismatches explicitly — if a RAG-retrieved condition lacks key symptom overlap, state it clearly (Critical Evaluation).
- Use precise clinical language appropriate for a physician audience.

## Medical Terminology Reference
General: Fatigue→อ่อนเพลีย | Weakness→อ่อนแรง | Malaise→ไม่สบายตัว | Chills→หนาวสั่น | Fever→ไข้ | High fever→ไข้สูง | Sweating→เหงื่อออก | Weight loss→น้ำหนักลด | Loss of appetite→เบื่ออาหาร
Neurological: Headache→ปวดหัว | Severe headache→ปวดหัวรุนแรง | Dizziness→เวียนหัว | Numbness→ชา | Blurred vision→มองเห็นไม่ชัด | Sensitivity to light→ไวต่อแสง (Photophobia) | Sensitivity to sound→ไวต่อเสียง (Phonophobia) | Pain behind eyes→ปวดหลังลูกตา
Respiratory: Cough→ไอ | Sputum→เสมหะ | Rusty sputum→เสมหะสีสนิม | Wheezing→หายใจมีเสียงหวีด | Shortness of breath→หายใจลำบาก | Chest tightness→แน่นหน้าอก | Chest pain→ปวดหน้าอก | Sore throat→เจ็บคอ | Runny nose→น้ำมูกไหล | Nasal congestion→คัดจมูก
Gastrointestinal: Nausea→คลื่นไส้ | Vomiting→อาเจียน | Abdominal pain→ปวดท้อง | Diarrhea→ท้องเสีย | Constipation→ท้องผูก | Excessive thirst→กระหายน้ำมาก | Glucose→กลูโคส/น้ำตาลในเลือด | Insulin→อินซูลิน
Urinary: Frequent urination→ปัสสาวะบ่อย | Burning urination→ปัสสาวะแสบขัด | Cloudy urine→ปัสสาวะขุ่น | Dark urine→ปัสสาวะสีเข้ม
Skin: Rash→ผื่น | Itching→คัน | Swelling→บวม | Red spots→จุดแดง | Blisters→ตุ่มน้ำ | Skin lesions→รอยโรคผิวหนัง | Jaundice (skin)→ผิวเหลือง
Musculoskeletal: Joint pain→ปวดข้อ | Muscle pain→ปวดกล้ามเนื้อ | Back pain→ปวดหลัง | Neck pain→ปวดคอ | Stiffness→ตึง/แข็ง | Weakness in limbs→แขนขาอ่อนแรง
"""

SEARCH_SYMPTOMS_PROMPT = """\
## Role
You are performing clinical differential diagnosis. Analyze the patient's symptoms against the retrieved clinical data and produce a structured diagnostic report.

## Patient Symptoms
{symptoms}

## Retrieved Clinical Data (RAG)
{rag}

---

## Output Instructions

### รายชื่อโรคที่เป็นไปได้ (Possible Conditions)
- Include ONLY conditions present in the RAG context with a real confidence score.
- Maximum 3 conditions. If RAG has fewer than 3, show only what is available — never invent rows.
- Sort strictly by confidence score: row 1 has the highest score.
- Format: Thai name first, English in parentheses.

| ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
|-------|-----------|-----------------|
| 1 | ชื่อภาษาไทย (English) | 0.XX |

Disease name mapping (mandatory — use exactly as shown):
Dengue→ไข้เลือดออก | Typhoid→ไข้ไทฟอยด์ | Pneumonia→ปอดอักเสบ | Diabetes→โรคเบาหวาน
Migraine→ไมเกรน | Malaria→มาลาเรีย | Allergy→ภูมิแพ้ | Common Cold→หวัดทั่วไป
Arthritis→โรคข้ออักเสบ | Hypertension→โรคความดันโลหิตสูง | Jaundice→ดีซ่าน
Chicken Pox→โรคอีสุกอีใส | Hepatitis A→โรคตับอักเสบ เอ | Psoriasis→โรคสะเก็ดเงิน
Impetigo→โรคพุพอง | Bronchial Asthma→โรคหอบหืด | Fungal Infection→การติดเชื้อเชื้อรา
Drug Reaction→การแพ้ยา | Urinary Tract Infection→โรคติดเชื้อทางเดินปัสสาวะ
Varicose Veins→เส้นเลือดขอด | Cervical Spondylosis→โรคกระดูกคอเสื่อม
Gastroesophageal Reflux Disease→โรคกรดไหลย้อน

### บทวิเคราะห์ทางคลินิก (Clinical Analysis)
For each condition in the table:
- State which reported symptoms support this diagnosis.
- State which key symptoms of this condition are absent from the patient's report (Critical Evaluation).
- Use precise clinical reasoning — avoid vague statements like "มีความเข้ากันได้".

### ข้อพิจารณาเพิ่มเติม (Clinical Considerations)
- Recommend specific next diagnostic steps: lab tests, imaging, or clinical history questions relevant to the top candidates.
- Highlight any red-flag symptoms that warrant urgent escalation.
- Keep recommendations concise and actionable.
"""

CONDITION_DETAILS_PROMPT = """\
## Role
You are summarizing the clinical profile of a specific condition for a physician. Use ONLY the information present in the RAG context below — do not add, infer, or fabricate any data.

## User Request
{query}

## Retrieved Clinical Data (RAG)
{rag}

---

## Output Instructions

### คำจำกัดความ (Definition)
Write 2–3 concise sentences describing what this condition is, based solely on the RAG context.

### อาการแสดงหลัก (Main Symptoms)
- List the most frequently reported symptoms from the RAG data (5–10 items).
- Sort by frequency of occurrence across patient cases.
- Format each item as: `N. ชื่ออาการภาษาไทย (English term)`
- Each symptom must have a unique Thai translation — never reuse the same Thai word for different symptoms.
- If a term cannot be translated accurately, use the English term in parentheses.

### Constraints
- Do NOT write sections on treatment, prevention, or clinical considerations unless the RAG context explicitly contains that information.
- Do NOT add any information beyond what is in the RAG context.
"""

TOOL_ROUTER_PROMPT = """\
You are a medical query router for a Clinical Decision Support System (CDSS).
Your task is to classify the user's query into exactly one of three routing targets based on intent.

## Routing Targets

### 1. get_condition_details
The user wants information about a specific, named medical condition.
- The query explicitly names a disease and asks about it (symptoms, definition, details, overview).
- The argument should be the English name of the condition.

### 2. search_symptoms
The user is describing observable physical symptoms that a patient is currently experiencing.
- The query contains concrete, clinical symptoms (e.g. fever, pain, rash, cough, swelling, vomiting).
- No specific disease name is mentioned — only symptoms.
- The argument should be an English translation of the symptoms.

### 3. none
Route here for everything that does not fit the above:
- Greetings, farewells, small talk, jokes.
- Vague or non-clinical complaints: "ผอม", "อ้วน", "เครียด", "ไม่สบาย" (without specific symptoms).
- Conditions outside the system's knowledge base (cancer, HIV, heart disease, etc.).
- Lifestyle, nutrition, mental health, or general wellness questions.
- Requests for treatment advice, prescriptions, or medical opinions.

## Query to Classify
{query}

Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.
Format: {{"tool": "<target>", "argument": "<value>"}}
"""


def _is_supported(name: str) -> bool:
    """Return True if name (English) matches any whitelisted condition."""
    n = name.lower().strip()
    if n in SUPPORTED_CONDITIONS:
        return True
    return any(s in n or n in s for s in SUPPORTED_CONDITIONS)


# Thai aliases — short/informal names that map to supported conditions
DISEASE_TH_ALIASES: dict[str, str] = {
    "ภูมิ": "allergy",
    "ภูมิแพ้": "allergy",
    "หวัด": "common cold",
    "ไข้หวัด": "common cold",
    "ไข้เลือดออก": "dengue",
    "เลือดออก": "dengue",
    "มาลาเรีย": "malaria",
    "ไข้มาลาเรีย": "malaria",
    "ไมเกรน": "migraine",
    "ปวดหัวไมเกรน": "migraine",
    "เบาหวาน": "diabetes",
    "น้ำตาล": "diabetes",
    "ความดัน": "hypertension",
    "ความดันโลหิต": "hypertension",
    "ปอดอักเสบ": "pneumonia",
    "ปอด": "pneumonia",
    "หอบหืด": "bronchial asthma",
    "หอบ": "bronchial asthma",
    "หืด": "bronchial asthma",
    "ไข้ไทฟอยด์": "typhoid",
    "ไทฟอยด์": "typhoid",
    "ดีซ่าน": "jaundice",
    "ตัวเหลือง": "jaundice",
    "ตับอักเสบ": "hepatitis a",
    "อีสุกอีใส": "chicken pox",
    "สุกใส": "chicken pox",
    "สะเก็ดเงิน": "psoriasis",
    "เส้นเลือดขอด": "varicose veins",
    "ข้ออักเสบ": "arthritis",
    "ข้อ": "arthritis",
    "กรดไหลย้อน": "gastroesophageal reflux disease",
    "กรด": "gastroesophageal reflux disease",
    "กระดูกคอ": "cervical spondylosis",
    "คอเสื่อม": "cervical spondylosis",
    "พุพอง": "impetigo",
    "ปัสสาวะ": "urinary tract infection",
    "ทางเดินปัสสาวะ": "urinary tract infection",
    "เชื้อรา": "fungal infection",
    "ราที่ผิวหนัง": "fungal infection",
    "แพ้ยา": "drug reaction",
}


# Clinical symptom keywords — query must contain at least one to be valid for search_symptoms
SYMPTOM_KEYWORDS = {
    # Thai symptoms
    "ไข้", "ปวด", "คัน", "บวม", "ผื่น", "ไอ", "เจ็บ", "หนาว", "สั่น", "เหนื่อย",
    "อ่อนเพลีย", "คลื่นไส้", "อาเจียน", "ท้องเสีย", "ท้องผูก", "หายใจ", "แน่น",
    "ชา", "เวียน", "ตา", "หู", "จมูก", "คอ", "หลัง", "ท้อง", "ขา", "แขน",
    "ปัสสาวะ", "อุจจาระ", "เลือด", "น้ำมูก", "เสมหะ", "ผิว", "ตุ่ม", "แผล",
    "กระหาย", "หิว", "อ่อนแรง", "ชาที่", "มองเห็น", "ปวดหัว", "ปวดท้อง",
    # English symptoms
    "fever", "pain", "ache", "rash", "cough", "itch", "swell", "nausea", "vomit",
    "diarrhea", "fatigue", "dizzy", "bleed", "discharge", "sore", "stiff", "numb",
    "breathe", "chest", "headache", "stomach", "urine", "skin", "spot", "blister",
    "thirst", "hunger", "weakness", "vision", "sweat", "chills", "sneeze",
}

# Thai disease names — if query contains these, it's a get_condition_details query, not search_symptoms
DISEASE_TH_NAMES = {
    "ไข้เลือดออก", "ไข้ไทฟอยด์", "ปอดอักเสบ", "เบาหวาน", "ไมเกรน", "มาลาเรีย",
    "ภูมิแพ้", "หวัด", "ข้ออักเสบ", "ความดัน", "กรดไหลย้อน", "สะเก็ดเงิน",
    "ดีซ่าน", "อีสุกอีใส", "ตับอักเสบ", "หอบหืด", "เส้นเลือดขอด", "พุพอง",
    "กระดูกคอเสื่อม", "แพ้ยา", "เชื้อรา", "ติดเชื้อ",
}


def _has_clinical_symptoms(text: str) -> bool:
    """Return True if text contains at least one recognizable clinical symptom keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SYMPTOM_KEYWORDS)


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

        if not API_KEY:
            raise EnvironmentError(
                "TYPHOON_API_KEY is not set. "
                "Please add it to your .env file (local) or HuggingFace Spaces Secrets (cloud)."
            )

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

        # Python-level guard: search_symptoms requires at least one clinical symptom keyword
        # Skip this guard if query contains a Thai disease name (should have been routed to get_condition_details)
        if tool_name == "search_symptoms":
            # Check aliases first (short/informal Thai names)
            matched_condition = None
            for alias, eng_name in DISEASE_TH_ALIASES.items():
                if alias in original:
                    matched_condition = eng_name
                    break
            # Also check DISEASE_TH_NAMES set
            if not matched_condition:
                for d in DISEASE_TH_NAMES:
                    if d in original:
                        matched_condition = d
                        break

            if matched_condition:
                logger.info(f"RE-ROUTE: disease name '{matched_condition}' detected → get_condition_details")
                tool_name = "get_condition_details"
                tool_input = {"condition_name": matched_condition}
            elif not _has_clinical_symptoms(original):
                logger.info(f"NO SYMPTOM KEYWORDS found in: '{original}' — rejecting")
                return (
                    "ขออภัยครับ ไม่พบอาการทางคลินิกที่ชัดเจนในคำถามของคุณ "
                    "กรุณาอธิบายอาการทางกายภาพให้ชัดเจนขึ้น เช่น มีไข้ ปวดหัว ไอ ผื่นขึ้น บวม ฯลฯ"
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
