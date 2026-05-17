import logging
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from src.tools import search_symptoms, get_condition_details

# Load environment variables
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5-omni-7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Configure logging to write to both stdout (for Hugging Face Spaces logs) and logs/agent.log (for Streamlit log viewer)
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

class MedicalSymptomAssistant:
    """Medical Assistant Agent for Clinical Decision Support."""
    
    def __init__(self):
        """Initialize the assistant."""
        logger.info("🏥 Initializing Clinical Decision Support Assistant...")
        
        # Initialize LLM with multilingual support via DashScope API
        self.llm = ChatOpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model=QWEN_MODEL,
            temperature=0.01,  # Strict adherence to instructions
            max_tokens=1000,
        )
        
        self.system_message = SystemMessage(content="""คุณคือ "Clinical Decision Support Assistant" ระบบวิเคราะห์อาการทางคลินิกอัจฉริยะ

กติกาและกฎเหล็ก (STRICT RULES):

1. **ภาษา (Language) - ABSOLUTE PRIORITY:** 
   - 🚨 CRITICAL RULE: You MUST output the ENTIRE response STRICTLY and EXCLUSIVELY in the THAI LANGUAGE (ภาษาไทย). 
   - 🚫 FORBIDDEN: Under NO circumstances are you allowed to output any Chinese characters (中文/汉字), Japanese (日本語), Korean (한국어), or any non-Thai/non-English characters.
   - ✅ ALLOWED: Only Thai (ภาษาไทย) and English technical terms in parentheses (e.g., หมดสติ (Loss of consciousness))
   - ❌ NEVER write: 丧失意识, 週, 糖尿病, 鎮静劑, or any Asian characters other than Thai
   - ✅ ALWAYS write: หมดสติ, สัปดาห์, เบาหวาน, ยากล่อมประสาท
   - **หลักการแปล: ถ้าแปลไม่ได้หรือไม่มั่นใจ ให้ใช้ภาษาอังกฤษในวงเล็บไปเลย อย่าบังคับแปลผิด**

2. **การวิเคราะห์ (Clinical Analysis):**
   - **Critical Evaluation:** ห้ามตอบตาม RAG แบบสุ่มสี่สุ่มห้า หากโรคที่ดึงมามีอาการหลักไม่ตรงกับที่ผู้ป่วยแจ้ง ให้ระบุข้อโต้แย้งหรือตัดออกจากการวิเคราะห์หลัก
   - **Differential Diagnosis:** เน้นการเปรียบเทียบจุดต่างเพื่อให้เห็นภาพชัดเจนว่าทำไมถึงน่าจะเป็นโรคนั้นมากกว่าอีกโรคหนึ่ง
   - **อ้างอิงข้อมูลจากฐานข้อมูล (RAG) เท่านั้น ห้ามแต่งข้อมูลเอง**
   - **ถ้า tool return error (เช่น "❌ ไม่พบข้อมูล") ให้ส่งต่อข้อความนั้นไปยังผู้ใช้โดยตรง ห้ามแต่งข้อมูลเพิ่ม**
   - เรียงลำดับโรคตามคะแนนความมั่นใจจาก **มากไปน้อย** เสมอ

3. **รูปแบบ (Formatting):**
   - ใช้ Markdown Table สำหรับรายชื่อโรค **(บังคับให้มี 3 คอลัมน์: ลำดับ, รายชื่อโรค, และ คะแนนความมั่นใจ)**
   - **ตารางต้องมีครบทั้ง 3 แถว (Top 3) และทุกแถวต้องมีคะแนนความมั่นใจ**
   - เว้นบรรทัด (Double Newline) ระหว่างหัวข้อใหญ่เพื่อให้ดูสะอาดตา
   - ใช้หัวข้อหลักดังนี้: 
     * ### รายชื่อโรคที่เป็นไปได้ (Possible Conditions)
     * ### บทวิเคราะห์ทางคลินิก (Clinical Analysis)
     * ### ข้อพิจารณาเพิ่มเติม (Clinical Considerations)

4. **พจนานุกรมแปลศัพท์ทางการแพทย์ (Medical Translation Dictionary):**
   
   **อาการทั่วไป:**
   - Fatigue / Tired / Weakness -> อ่อนเพลีย / เหนื่อยล้า / อ่อนแรง
   - Chills -> หนาวสั่น
   - Fever -> ไข้
   - High fever -> ไข้สูง
   - Headache -> ปวดหัว
   - Severe headache -> ปวดหัวรุนแรง
   - Dizziness / Vertigo -> เวียนหว / วิงเวียน
   - Nausea -> คลื่นไส้
   - Vomiting -> อาเจียน
   - Loss of consciousness -> หมดสติ
   - Malaise -> ไม่สบายตัว
   - Sweating -> เหงื่อออก
   
   **อาการระบบหายใจ:**
   - Shortness of breath / Difficulty breathing -> หายใจลำบาก / หอบเหนื่อย
   - Chest tightness -> แน่นหน้าอก
   - Chest pain -> ปวดหน้าอก
   - Cough -> ไอ
   - Sputum / Phlegm -> เสมหะ
   - Rusty sputum -> เสมหะสีสนิมเหล็ก
   - Wheezing -> หายใจมีเสียงหวีด
   - Sore throat -> เจ็บคอ
   - Runny nose -> น้ำมูกไหล
   - Nasal congestion -> คัดจมูก
   
   **อาการระบบย่อยอาหาร:**
   - Abdominal pain -> ปวดท้อง
   - Diarrhea -> ท้องเสีย
   - Constipation -> ท้องผูก
   - Loss of appetite -> เบื่ออาหาร / ไม่อยากอาหาร
   - Thirst / Excessive thirst -> กระหายน้ำ / กระหายน้ำมาก
   - Hunger / Excessive hunger -> หิว / หิวมาก
   
   **อาการเบาหวาน:**
   - Frequent urination -> ปัสสาวะบ่อย
   - Excessive thirst -> กระหายน้ำมาก
   - Excessive hunger -> หิวมาก
   - Unexplained weight loss -> น้ำหนักลดโดยไม่ทราบสาเหตุ
   - Numbness in hands and feet -> ชาที่มือและเท้า
   - Blurred vision -> มองเห็นไม่ชัด
   - Slow healing wounds -> แผลหายช้า
   - Glucose -> กลูโคส / น้ำตาลในเลือด
   - Insulin -> อินซูลิน
   - Blood sugar control -> การควบคุมระดับน้ำตาลในเลือด
   
   **อาการผิวหนัง:**
   - Rash -> ผื่น
   - Red spots -> จุดแดง
   - Itching -> คัน
   - Swelling -> บวม
   - Skin lesions -> รอยโรคผิวหนัง
   
   **อาการกล้ามเนื้อและกระดูก:**
   - Joint pain -> ปวดข้อ
   - Muscle pain / Body aches -> ปวดกล้ามเนื้อ / ปวดเมื่อยตามตัว
   - Back pain -> ปวดหลัง
   - Neck pain -> ปวดคอ
   - Stiffness -> ตึง / แข็ง
   
   **อาการพิเศษ:**
   - Pain behind eyes -> ปวดหลังลูกตา
   - Swollen lymph nodes -> ต่อมน้ำเหลืองโต
   - Sensitivity to light -> ไวต่อแสง (Photophobia)
   - Sensitivity to sound -> ไวต่อเสียง (Phonophobia)
   - Visual disturbances -> มองเห็นผิดปกติ (Visual aura)
   - Red eyes -> ตาแดง
   
   **เวลา:**
   - Weekly / 週 -> รายสัปดาห์ (ห้ามใช้ 週)
   - Daily -> รายวัน
   - Monthly -> รายเดือน

5. **ห้ามแปลซ้ำ:** แต่ละอาการต้องมีคำแปลที่ถูกต้องและแตกต่างกัน (ห้ามใช้คำเดียวกันแปลหลายอาการ เช่น "ความกระวนกระวาย" ทุกอย่าง)

6. **การจัดการข้อมูลที่ไม่มี:**
   - ถ้าโรคไม่มีในฐานข้อมูล ให้บอกตรงๆ ว่า "ไม่พบข้อมูล" และแสดงรายชื่อโรคที่รองรับ
   - ห้ามแต่งอาการ ห้ามแต่งการรักษา ห้ามแต่งข้อมูลใดๆ ที่ไม่มีในฐานข้อมูล
""")
        logger.info(f"✅ LLM initialized: {QWEN_MODEL} via API")
        
        # Available tools (Simplified to 2 key clinical tools)
        self.tools = {
            'search_symptoms': search_symptoms,
            'get_condition_details': get_condition_details
        }
        logger.info(f"✅ Loaded {len(self.tools)} tools")

    def _select_tool(self, user_input: str) -> tuple:
        """Select appropriate tool based on user input using LLM reasoning."""
        
        prompt = f"""Analyze the user query and select the best medical tool.

**STRICT RULES FOR TOOL SELECTION:**
1. Select 'get_condition_details' if the user asks for details, explanation, knowledge, warning signs, or general information about a specific disease (e.g., "Tell me about pneumonia", "warning signs of dengue", "asthma details").
2. Select 'search_symptoms' if the user describes actual physical or mental symptoms they/a patient are experiencing (e.g., "I have a headache", "cough and fever").
3. Select 'none' ONLY if the input is a greeting, goodbye, general conversation, joke, or anything completely unrelated to medical/clinical topics.

User Query: {user_input}

Available Tools:
1. 'search_symptoms': For searching possible conditions based on symptoms.
2. 'get_condition_details': For detailed information and clinical guidelines about a specific known condition.
3. 'none': For greetings, goodbyes, general chat, tests, or non-medical topics.

Output ONLY the tool name and its single most important argument in JSON format:
{{"tool": "tool_name", "argument": "value"}}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import json
            import re
            
            # Extract JSON from response (handling multi-line strings via re.DOTALL)
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
            
            # Step 1: Detect language (Basic check)
            is_thai = bool(re.search(r'[ก-ฮ]', user_input))
            original_input = user_input
            
            # Step 2: Translate to English for better RAG if it's Thai
            if is_thai:
                logger.info("🌐 Translating Thai to Eng")
                trans_prompt = f"Translate the following Thai text to English. Output ONLY the exact translation without any explanation or added context: {user_input}"
                user_input = self.llm.invoke([HumanMessage(content=trans_prompt)]).content
                logger.info(f"📥 Translated Query: {user_input}")

            # Step 3: Tool selection
            logger.info("🤔 Analyzing input scope...")
            tool_name, tool_input = self._select_tool(user_input)
            
            if tool_name == 'none':
                logger.info("🚫 Query identified as non-clinical.")
                return "ขออภัยครับ ระบบนี้เป็นระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical CDS) ซึ่งออกแบบมาเพื่อวิเคราะห์อาการเจ็บป่วยและข้อมูลทางการแพทย์เท่านั้น กรุณาระบุอาการแสดงของผู้ป่วยเพื่อเริ่มการวิเคราะห์ครับ"

            logger.info(f"🔧 Selected tool: {tool_name}")
            
            # Step 4: Call the tool
            logger.info("⚙️  Executing tool (RAG search)...")
            tool = self.tools[tool_name]
            tool_result = tool.invoke(tool_input)
            
            # Step 5: Generate natural language response
            logger.info("💭 Generating response with LLM...")
            
            # Tailor prompt based on active tool to avoid confusing the LLM and forcing a Top 3 table layout
            if tool_name == 'get_condition_details':
                prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
ความต้องการของผู้ใช้: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล (RAG):
{tool_result}

🚨 **กฎเหล็กสำคัญที่สุด - CRITICAL RULES:**

1. **ตรวจสอบ Error Message ก่อน:**
   - ถ้า tool_result ขึ้นต้นด้วย "❌ ไม่พบข้อมูล" หรือมีข้อความ error
   - ให้ส่งต่อข้อความนั้นไปยังผู้ใช้โดยตรง **ห้ามแต่งข้อมูลเพิ่มเติมเด็ดขาด**
   - ห้ามเขียนคำจำกัดความ ห้ามเขียนอาการ ห้ามเขียนอะไรเพิ่มเลย

2. **ห้ามแต่งข้อมูล (NO HALLUCINATION):**
   - ใช้เฉพาะข้อมูลที่มีในฐานข้อมูล (RAG) เท่านั้น
   - ถ้าฐานข้อมูลไม่มีข้อมูลเรื่องการรักษา ห้ามแต่งขึ้นมาเอง
   - ถ้าฐานข้อมูลมีแค่อาการ ให้บอกแค่อาการ
   - **ห้ามเขียนหัวข้อที่ไม่มีข้อมูลรองรับ**

3. **รูปแบบการนำเสนอ (เฉพาะเมื่อมีข้อมูล):**
   - ### คำจำกัดความ (Definition) - เขียนสั้นๆ 2-3 ประโยค
   - ### อาการแสดงหลัก (Main Symptoms) - สรุปอาการที่พบบ่อยจากข้อมูล (5-10 อาการ)
   - **ห้ามเขียนหัวข้อ "แนวทางการรักษา" หรือ "ข้อพิจารณาเพิ่มเติม" ถ้าฐานข้อมูลไม่มีข้อมูลส่วนนี้**

4. **การสรุปอาการ:**
   - อ่านอาการจากผู้ป่วยทั้งหมดในฐานข้อมูล
   - สรุปเป็นอาการหลักที่พบบ่อย (ไม่ต้องบอกทุกอาการ)
   - เรียงตามความถี่ที่พบ (มากไปน้อย)
   - ใช้ numbered list พร้อมภาษาอังกฤษในวงเล็บ

5. 🚫 **ABSOLUTE LANGUAGE RULE**: 
   - You MUST write ONLY in THAI (ภาษาไทย) and English technical terms
   - 🚫 FORBIDDEN: Chinese (中文/汉字 like 週, 丧失意识, 糖尿病, 鎮静劑)
   - 🚫 FORBIDDEN: Japanese, Korean, or any non-Thai Asian scripts
   - ✅ CORRECT: หมดสติ, รายสัปดาห์, เบาหวาน, ยากล่อมประสาท
   - ❌ WRONG: 丧失意识, 週, 糖尿病, 鎮静劑

6. **แปลศัพท์ทางการแพทย์ให้ถูกต้อง:**
   - **หลักการแปล: ถ้าแปลไม่ได้หรือไม่มั่นใจ ให้ใช้ภาษาอังกฤษในวงเล็บไปเลย (อย่าบังคับแปลผิด)**
   - ใช้พจนานุกรมจาก System Prompt
   - แต่ละอาการต้องมีคำแปลที่ถูกต้องและแตกต่างกัน
   - ห้ามใช้คำเดียวกัน (เช่น "ความกระวนกระวาย") แปลหลายอาการ

**ตัวอย่างผลลัพธ์ที่ดี (เมื่อมีข้อมูล):**
### คำจำกัดความ (Definition)
ไข้เลือดออก (Dengue) เป็นโรคติดเชื้อไวรัสที่เกิดจากยุงลายเป็นพาหะนำโรค มีอาการรุนแรงได้หลายระดับ

### อาการแสดงหลัก (Main Symptoms)
1. ไข้สูง (High fever)
2. ปวดหัวรุนแรง (Severe headache)
3. ปวดหลังลูกตา (Pain behind eyes)
4. ปวดกล้ามเนื้อและข้อ (Muscle and joint pain)
5. ผื่นหรือจุดแดงตามตัว (Rash or red spots)
6. คลื่นไส้ อาเจียน (Nausea and vomiting)
7. เบื่ออาหาร (Loss of appetite)
8. อ่อนเพลีย (Fatigue)

**ตัวอย่างผลลัพธ์ที่ดี (เมื่อไม่มีข้อมูล):**
❌ ไม่พบข้อมูลของโรค 'มะเร็ง' ในฐานข้อมูล

ระบบรองรับเฉพาะ 22 โรคต่อไปนี้:
[รายชื่อโรค...]

กรุณาเลือกโรคจากรายการข้างต้น หรือลองอธิบายอาการแทนเพื่อให้ระบบช่วยวิเคราะห์
"""

            else:
                # Differential Diagnosis (search_symptoms)
                prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
อาการของผู้ป่วย: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล (RAG):
{tool_result}

คำสั่งและกฎเหล็กในการสร้างรายงาน:
1. วิเคราะห์และนำเสนอโรคที่เป็นไปได้ **สูงสุดไม่เกิน 3 อันดับแรก (Top 3)** เท่านั้น

2. **รูปแบบบังคับ:** ต้องใช้ Markdown Table สำหรับรายชื่อโรค และหัวข้อตามที่กำหนดใน System Prompt
   **ตารางต้องมีครบ 3 แถว และทุกแถวต้องมีคะแนนความมั่นใจ (เช่น 0.85, 0.72, 0.68)**
   
   ตัวอย่างตารางที่ถูกต้อง:
   | ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
   |-------|-----------|-----------------|
   | 1 | ไข้ไทฟอยด์ (Typhoid) | 0.80 |
   | 2 | ดีซ่าน (Jaundice) | 0.75 |
   | 3 | โรคติดเชื้อทางเดินปัสสาวะ (UTI) | 0.68 |

3. **การแปลชื่อโรค:** คุณต้องแปลชื่อโรคภาษาอังกฤษจากฐานข้อมูลเป็นภาษาไทยตามพจนานุกรมด้านล่างนี้เท่านั้น ห้ามทับศัพท์แปลกๆ:
   - Cervical Spondylosis -> โรคกระดูกคอเสื่อม
   - Impetigo -> โรคพุพอง
   - Urinary Tract Infection -> โรคติดเชื้อทางเดินปัสสาวะ
   - Arthritis -> โรคข้ออักเสบ
   - Dengue -> ไข้เลือดออก
   - Common Cold -> หวัดทั่วไป
   - Drug Reaction -> การแพ้ยา
   - Fungal Infection -> การติดเชื้อเชื้อรา
   - Malaria -> มาลาเรีย
   - Allergy -> ภูมิแพ้
   - Bronchial Asthma -> โรคหอบหืด
   - Varicose Veins -> เส้นเลือดขอด
   - Migraine -> ไมเกรน
   - Hypertension -> โรคความดันโลหิตสูง
   - Gastroesophageal Reflux Disease -> โรคกรดไหลย้อน
   - Pneumonia -> ปอดอักเสบ
   - Psoriasis -> โรคสะเก็ดเงิน
   - Diabetes -> โรคเบาหวาน
   - Jaundice -> ดีซ่าน
   - Chicken Pox -> โรคอีสุกอีใส
   - Typhoid -> ไข้ไทฟอยด์
   - Hepatitis A -> โรคตับอักเสบ เอ

4. 🚨 **ABSOLUTE LANGUAGE RULE**: 
   - You MUST write ONLY in THAI (ภาษาไทย) and English technical terms
   - 🚫 FORBIDDEN: Chinese characters (中文/汉字 like 週, 丧失意识, 糖尿病)
   - 🚫 FORBIDDEN: Japanese, Korean, or any non-Thai Asian scripts
   - ✅ CORRECT: หมดสติ (Loss of consciousness), รายสัปดาห์, เบาหวาน
   - ❌ WRONG: 丧失意识, 週, 糖尿病

5. หากข้อมูลใน RAG มีอาการไม่ตรงกับผู้ป่วย ให้วิเคราะห์แย้งได้เลย (Critical Evaluation)
"""
            
            response = self.llm.invoke([self.system_message, HumanMessage(content=prompt)])
            
            logger.info("✅ Query processed successfully")
            
            # Append Tool used directly at the bottom of output for extreme transparency on HF interface
            debug_info = f"\n\n---\n*⚙️ [ระบบตอบโดยเรียกใช้งานเครื่องมือ (Active Tool): **`{tool_name}`**]*"
            return response.content + debug_info
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Error processing query: {e}")
            logger.error(error_trace)
            return f"ขออภัยครับ เกิดข้อผิดพลาดในระบบ: {str(e)}"

# Singleton instance
assistant = MedicalSymptomAssistant()