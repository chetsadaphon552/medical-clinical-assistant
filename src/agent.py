import json
import logging
import os
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.tools import search_symptoms, get_condition_details

# Load environment variables
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5-omni-7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medical_assistant")

# ---------------------------------------------------------------------------
# Hard-coded whitelist: only these 22 conditions exist in the vector store.
# Any query about a condition NOT in this set is rejected BEFORE calling LLM.
# ---------------------------------------------------------------------------
SUPPORTED_CONDITIONS = {
    'cervical spondylosis', 'impetigo', 'urinary tract infection', 'arthritis',
    'dengue', 'common cold', 'drug reaction', 'fungal infection', 'malaria',
    'allergy', 'bronchial asthma', 'varicose veins', 'migraine', 'hypertension',
    'gastroesophageal reflux disease', 'pneumonia', 'psoriasis', 'diabetes',
    'jaundice', 'chicken pox', 'typhoid', 'hepatitis a'
}

NOT_FOUND_MSG = """ ไม่พบข้อมูลของโรคนี้ในฐานข้อมูล

ระบบรองรับเฉพาะ **22 โรค** ต่อไปนี้เท่านั้น:

| ลำดับ | รายชื่อโรค (ภาษาไทย) | English Name |
|-------|---------------------|--------------|
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

💡 **คำแนะนำ:** กรุณาเลือกโรคจากรายการข้างต้น หรือลองอธิบายอาการของผู้ป่วยแทน เพื่อให้ระบบช่วยวิเคราะห์"""


def _is_supported_condition(condition_name: str) -> bool:
    """Check if a condition name (English) is in the supported whitelist."""
    name = condition_name.lower().strip()
    # Exact match
    if name in SUPPORTED_CONDITIONS:
        return True
    # Partial match (e.g. "dengue fever" -> "dengue")
    for supported in SUPPORTED_CONDITIONS:
        if supported in name or name in supported:
            return True
    return False


class MedicalSymptomAssistant:
    """Medical Assistant Agent for Clinical Decision Support."""

    def __init__(self):
        logger.info("🏥 Initializing Clinical Decision Support Assistant...")

        self.llm = ChatOpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model=QWEN_MODEL,
            temperature=0.01,
            max_tokens=1000,
        )

        self.system_message = SystemMessage(content="""คุณคือ "Clinical Decision Support Assistant" ระบบวิเคราะห์อาการทางคลินิกอัจฉริยะ

กติกาและกฎเหล็ก (STRICT RULES):

1. **ภาษา (Language) - ABSOLUTE PRIORITY:**
   - 🚨 CRITICAL RULE: You MUST output the ENTIRE response STRICTLY and EXCLUSIVELY in the THAI LANGUAGE (ภาษาไทย).
   - 🚫 FORBIDDEN: Chinese (中文/汉字), Japanese (日本語), Korean (한국어), or any non-Thai/non-English characters.
   - ✅ ALLOWED: Thai + English technical terms in parentheses only
   - ❌ NEVER write: 丧失意识, 週, 糖尿病, 鎮静劑
   - **หลักการแปล: ถ้าแปลไม่ได้หรือไม่มั่นใจ ให้ใช้ภาษาอังกฤษในวงเล็บไปเลย อย่าบังคับแปลผิด**

2. **การวิเคราะห์ (Clinical Analysis):**
   - อ้างอิงข้อมูลจากฐานข้อมูล (RAG) เท่านั้น ห้ามแต่งข้อมูลเอง
   - ถ้า RAG return error ให้ส่งต่อข้อความนั้นตรงๆ ห้ามแต่งข้อมูลเพิ่ม
   - เรียงลำดับโรคตามคะแนนความมั่นใจจากมากไปน้อย

3. **รูปแบบ (Formatting) สำหรับ search_symptoms:**
   - ใช้ Markdown Table 3 คอลัมน์: ลำดับ, รายชื่อโรค, คะแนนความมั่นใจ
   - ตารางต้องมีครบ 3 แถว ทุกแถวต้องมีคะแนน
   - หัวข้อ: ### รายชื่อโรคที่เป็นไปได้, ### บทวิเคราะห์ทางคลินิก, ### ข้อพิจารณาเพิ่มเติม

4. **พจนานุกรมแปลศัพท์ทางการแพทย์:**
   - Fatigue/Weakness -> อ่อนเพลีย/อ่อนแรง | Chills -> หนาวสั่น | Fever -> ไข้
   - Headache -> ปวดหัว | Dizziness -> เวียนหว | Nausea -> คลื่นไส้ | Vomiting -> อาเจียน
   - Shortness of breath -> หายใจลำบาก | Chest tightness -> แน่นหน้าอก | Cough -> ไอ
   - Sputum -> เสมหะ | Wheezing -> หายใจมีเสียงหวีด | Sore throat -> เจ็บคอ
   - Abdominal pain -> ปวดท้อง | Diarrhea -> ท้องเสีย | Loss of appetite -> เบื่ออาหาร
   - Frequent urination -> ปัสสาวะบ่อย | Excessive thirst -> กระหายน้ำมาก
   - Numbness -> ชา | Blurred vision -> มองเห็นไม่ชัด | Weight loss -> น้ำหนักลด
   - Rash -> ผื่น | Itching -> คัน | Swelling -> บวม | Red spots -> จุดแดง
   - Joint pain -> ปวดข้อ | Muscle pain -> ปวดกล้ามเนื้อ | Back pain -> ปวดหลัง
   - Pain behind eyes -> ปวดหลังลูกตา | Sensitivity to light -> ไวต่อแสง (Photophobia)
   - Sensitivity to sound -> ไวต่อเสียง (Phonophobia) | Sweating -> เหงื่อออก
   - Weekly/週 -> รายสัปดาห์ | Glucose -> กลูโคส/น้ำตาลในเลือด | Insulin -> อินซูลิน

5. **ห้ามแปลซ้ำ:** แต่ละอาการต้องมีคำแปลที่ถูกต้องและแตกต่างกัน
""")

        self.tools = {
            'search_symptoms': search_symptoms,
            'get_condition_details': get_condition_details
        }
        logger.info(f"✅ LLM initialized: {QWEN_MODEL} | Tools: {len(self.tools)}")

    def _select_tool(self, user_input: str) -> tuple:
        """Select appropriate tool based on user input using LLM reasoning."""
        prompt = f"""You are a medical tool router. Analyze the user query and select the correct tool.

RULES (follow strictly in order):
1. Use 'get_condition_details' when:
   - User asks about symptoms OF a specific named disease (e.g. "symptoms of dengue", "what is malaria", "tell me about diabetes")
   - User asks for details/info/explanation about a specific disease by name
   - Keywords: อาการของ, อาการโรค, รายละเอียด, บอกเกี่ยวกับ, เป็นยังไง, symptoms of, tell me about, what is, details of

2. Use 'search_symptoms' when:
   - User describes symptoms they/patient are CURRENTLY EXPERIENCING
   - No specific disease name is mentioned, only symptoms
   - Keywords: มีไข้, ปวด, คัน, เหนื่อย, I have, I feel, patient has

3. Use 'none' when:
   - Greeting, goodbye, joke, or completely non-medical topic

IMPORTANT: If the query contains a disease name AND asks about its symptoms/details → use 'get_condition_details'
IMPORTANT: If the query only describes physical symptoms without naming a disease → use 'search_symptoms'

Examples:
- "อาการไข้เลือดออก" → get_condition_details (argument: "dengue")
- "โรคตับอักเสบมีอาการอะไร" → get_condition_details (argument: "hepatitis a")
- "โรคเบาหวานเป็นยังไง" → get_condition_details (argument: "diabetes")
- "มีไข้ ปวดหัว อ่อนเพลีย" → search_symptoms
- "ปวดขา บวม" → search_symptoms
- "สวัสดี" → none

User Query: {user_input}

Output ONLY JSON: {{"tool": "tool_name", "argument": "value"}}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                tool = data.get('tool', 'search_symptoms')
                arg = data.get('argument', user_input)
                if tool == 'get_condition_details':
                    return 'get_condition_details', {'condition_name': arg}
                elif tool == 'none':
                    return 'none', None
                else:
                    return 'search_symptoms', {'symptoms': arg}
            return 'search_symptoms', {'symptoms': user_input}
        except Exception as e:
            logger.error(f"Error selecting tool: {e}")
            return 'search_symptoms', {'symptoms': user_input}

    def query(self, user_input: str) -> str:
        """Process user query and return response."""
        try:
            logger.info("=" * 80)
            logger.info(f"👤 User Query: {user_input}")
            logger.info("=" * 80)

            is_thai = bool(re.search(r'[ก-ฮ]', user_input))
            original_input = user_input

            # Translate Thai -> English for better RAG retrieval
            if is_thai:
                logger.info("🌐 Translating Thai to English...")
                trans_prompt = f"Translate the following Thai text to English. Output ONLY the exact translation without any explanation: {user_input}"
                user_input = self.llm.invoke([HumanMessage(content=trans_prompt)]).content
                logger.info(f"📥 Translated: {user_input}")

            # Tool selection
            logger.info("🤔 Selecting tool...")
            tool_name, tool_input = self._select_tool(user_input)

            if tool_name == 'none':
                logger.info("🚫 Non-clinical query rejected.")
                return "ขออภัยครับ ระบบนี้เป็นระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical CDS) ออกแบบมาเพื่อวิเคราะห์อาการเจ็บป่วยและข้อมูลทางการแพทย์เท่านั้น กรุณาระบุอาการแสดงของผู้ป่วยเพื่อเริ่มการวิเคราะห์ครับ"

            # ---------------------------------------------------------------
            # HARD VALIDATION for get_condition_details:
            # Check whitelist BEFORE calling LLM or tool.
            # This prevents hallucination on unknown diseases entirely.
            # ---------------------------------------------------------------
            if tool_name == 'get_condition_details':
                condition_arg = tool_input.get('condition_name', '')
                if not _is_supported_condition(condition_arg):
                    logger.info(f"🚫 Condition '{condition_arg}' not in whitelist. Returning NOT_FOUND_MSG.")
                    debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
                    return NOT_FOUND_MSG + debug_info

            logger.info(f"🔧 Tool: {tool_name} | Input: {tool_input}")

            # Call the tool
            logger.info("⚙️ Executing tool (RAG search)...")
            tool_fn = self.tools[tool_name]
            tool_result = tool_fn.invoke(tool_input)

            # If tool itself returned a not-found message, return it directly
            if tool_result.startswith("❌"):
                logger.info("🚫 Tool returned not-found. Returning directly.")
                debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
                return tool_result + debug_info

            # Post-filter search_symptoms results: remove conditions not in whitelist
            if tool_name == 'search_symptoms':
                filtered_lines = []
                skip_block = False
                for line in tool_result.split('\n'):
                    cond_match = re.match(r'^Condition:\s*(.+)', line, re.IGNORECASE)
                    if cond_match:
                        cond_name = cond_match.group(1).strip()
                        if _is_supported_condition(cond_name):
                            skip_block = False
                            filtered_lines.append(line)
                        else:
                            skip_block = True
                            logger.info(f"🚫 Filtered out unsupported condition: {cond_name}")
                    elif not skip_block:
                        filtered_lines.append(line)
                tool_result = '\n'.join(filtered_lines)
                if not tool_result.strip():
                    debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
                    return NOT_FOUND_MSG + debug_info

                # Check confidence scores - reject if max score too low (non-clinical / irrelevant query)
                scores = re.findall(r'Confidence Score:\s*([\d.]+)', tool_result)
                if scores:
                    max_score = max(float(s) for s in scores)
                    if max_score < 0.62:
                        logger.info(f"🚫 Max confidence {max_score:.2f} too low - likely non-clinical query.")
                        debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
                        return "ขออภัยครับ ไม่พบโรคในฐานข้อมูลที่สอดคล้องกับอาการที่ระบุ กรุณาอธิบายอาการทางกายภาพให้ชัดเจนขึ้น เช่น มีไข้ ปวดหัว ไอ ผื่นขึ้น ฯลฯ" + debug_info

            # Generate LLM response
            logger.info("💭 Generating LLM response...")

            if tool_name == 'get_condition_details':
                prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
ความต้องการของผู้ใช้: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล (RAG):
{tool_result}

🚨 CRITICAL RULES:
1. ใช้เฉพาะข้อมูลที่มีใน RAG เท่านั้น ห้ามแต่งข้อมูลเพิ่มเด็ดขาด
2. รูปแบบ (เฉพาะเมื่อมีข้อมูล):
   - ### คำจำกัดความ (Definition) - 2-3 ประโยค
   - ### อาการแสดงหลัก (Main Symptoms) - 5-10 อาการที่พบบ่อย เรียงตามความถี่
   - ห้ามเขียนหัวข้อ "แนวทางการรักษา" หรือ "ข้อพิจารณาเพิ่มเติม" ถ้าไม่มีข้อมูล
3. ใช้ numbered list พร้อมภาษาอังกฤษในวงเล็บ เช่น: 1. ไข้สูง (High fever)
4. ถ้าแปลศัพท์ไม่ได้ ให้ใช้ภาษาอังกฤษในวงเล็บไปเลย อย่าบังคับแปลผิด
5. ห้ามภาษาจีน/ญี่ปุ่น/เกาหลี ใช้ภาษาไทย + อังกฤษเท่านั้น
6. ห้ามแปลซ้ำ แต่ละอาการต้องมีคำแปลที่แตกต่างกัน
"""
            else:
                prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
อาการของผู้ป่วย: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล (RAG):
{tool_result}

🚨 CRITICAL RULES:
1. นำเสนอโรคที่เป็นไปได้ **สูงสุด 3 อันดับแรก (Top 3)** เท่านั้น โดยใช้เฉพาะโรคที่มีใน RAG ข้างต้น ห้ามเพิ่มโรคอื่นเด็ดขาด

2. **ตารางบังคับ** - ต้องครบ 3 แถว เรียงคะแนนจากมากไปน้อย (แถว 1 > แถว 2 > แถว 3):
   | ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
   |-------|-----------|-----------------|
   | 1 | ชื่อภาษาไทย (English) | 0.XX |
   | 2 | ชื่อภาษาไทย (English) | 0.XX |
   | 3 | ชื่อภาษาไทย (English) | 0.XX |

3. **บังคับแปลชื่อโรคเป็นภาษาไทยในตาราง** (ห้ามใส่แค่ภาษาอังกฤษ):
   Dengue→ไข้เลือดออก, Typhoid→ไข้ไทฟอยด์, Pneumonia→ปอดอักเสบ,
   Diabetes→โรคเบาหวาน, Migraine→ไมเกรน, Malaria→มาลาเรีย, Allergy→ภูมิแพ้,
   Common Cold→หวัดทั่วไป, Arthritis→โรคข้ออักเสบ, Hypertension→โรคความดันโลหิตสูง,
   Jaundice→ดีซ่าน, Chicken Pox→โรคอีสุกอีใส, Hepatitis A→โรคตับอักเสบ เอ,
   Psoriasis→โรคสะเก็ดเงิน, Impetigo→โรคพุพอง, Bronchial Asthma→โรคหอบหืด,
   Fungal Infection→การติดเชื้อเชื้อรา, Drug Reaction→การแพ้ยา,
   Urinary Tract Infection→โรคติดเชื้อทางเดินปัสสาวะ, Varicose Veins→เส้นเลือดขอด,
   Cervical Spondylosis→โรคกระดูกคอเสื่อม, Gastroesophageal Reflux Disease→โรคกรดไหลย้อน

4. 🚫 ห้ามภาษาจีน/ญี่ปุ่น/เกาหลี/เวียดนาม ใช้ภาษาไทย + อังกฤษเท่านั้น
   ห้ามใช้เครื่องหมาย 。% ที่ไม่ใช่ภาษาไทย/อังกฤษ

5. หัวข้อ: ### รายชื่อโรคที่เป็นไปได้, ### บทวิเคราะห์ทางคลินิก, ### ข้อพิจารณาเพิ่มเติม

6. หากอาการใน RAG ไม่ตรงกับผู้ป่วย ให้วิเคราะห์แย้งได้ (Critical Evaluation)
"""

            response = self.llm.invoke([self.system_message, HumanMessage(content=prompt)])
            logger.info("✅ Query processed successfully")

            debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
            return response.content + debug_info

        except Exception as e:
            import traceback
            logger.error(f"❌ Error: {e}\n{traceback.format_exc()}")
            return f"ขออภัยครับ เกิดข้อผิดพลาดในระบบ: {str(e)}"


# Singleton instance
assistant = MedicalSymptomAssistant()
