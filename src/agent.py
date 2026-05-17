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
1. **ภาษา (Language):** 
   - CRITICAL RULE: You MUST output the ENTIRE response STRICTLY and EXCLUSIVELY in the THAI LANGUAGE (ภาษาไทย). 
   - Under NO circumstances are you allowed to output any Chinese (中文) or other languages. Think and write ONLY in Thai.
   - ห้ามมีตัวอักษรภาษาจีนหรือคำศัพท์จีนหลุดออกมาในผลลัพธ์โดยเด็ดขาด!
   - แปลศัพท์เทคนิคให้เป็นภาษาไทยมาตรฐาน (เช่น Sputum -> เสมหะ, Rusty -> สีสนิมเหล็ก, Malaise -> ไม่สบายเนื้อไม่สบายตัว, Loss of consciousness -> หมดสติ)

2. **การวิเคราะห์ (Clinical Analysis):**
   - **Critical Evaluation:** ห้ามตอบตาม RAG แบบสุ่มสี่สุ่มห้า หากโรคที่ดึงมามีอาการหลักไม่ตรงกับที่ผู้ป่วยแจ้ง (เช่น RAG บอกว่ามีจามในโรคแพ้ยา แต่ผู้ป่วยไม่ได้แจ้งประวัติการใช้ยา) ให้ AI ระบุข้อโต้แย้งหรือตัดออกจากการวิเคราะห์หลัก
   - **Differential Diagnosis:** เน้นการเปรียบเทียบจุดต่างเพื่อให้เห็นภาพชัดเจนว่าทำไมถึงน่าจะเป็นโรคนั้นมากกว่าอีกโรคหนึ่ง
   - อ้างอิงข้อมูลจากฐานข้อมูล (RAG) เท่านั้น ห้ามแต่งข้อมูลเอง
   - เรียงลำดับโรคตามคะแนนความมั่นใจจาก **มากไปน้อย** เสมอ

3. **รูปแบบ (Formatting):**
   - ใช้ Markdown Table สำหรับรายชื่อโรค **(บังคับให้มี 3 คอลัมน์: ลำดับ, รายชื่อโรค, และ คะแนนความมั่นใจ)**
   - เว้นบรรทัด (Double Newline) ระหว่างหัวข้อใหญ่เพื่อให้ดูสะอาดตา
   - ใช้หัวข้อหลักดังนี้: ### รายชื่อโรคที่เป็นไปได้ (Possible Conditions), ### บทวิเคราะห์ทางคลินิก (Clinical Analysis), ### ข้อพิจารณาเพิ่มเติม (Clinical Considerations)

     *ตารางพจนานุกรมแปลศัพท์ (Internal Translation Map):*
     - Chest tightness -> แน่นหน้าอก
     - Shortness of breath / Difficulty breathing -> หายใจลำบาก / หอบเหนื่อย
     - Fatigue / Always tired -> อ่อนเพลีย / เหนื่อยล้าตลอดเวลา
     - Chills -> หนาวสั่น
     - Common Cold -> หวัดทั่วไป
     - Pneumonia -> ปอดอักเสบ
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

คำสั่งและกฎเหล็กในการสร้างรายงาน:
1. นำเสนอข้อมูลรายละเอียดเชิงลึกของโรคที่ผู้ใช้สอบถามโดยอิงตามข้อมูลจากฐานข้อมูลทางการแพทย์ที่ให้มาเท่านั้น ห้ามแต่งข้อมูลขึ้นมาเองเด็ดขาด
2. เขียนอธิบายแบ่งเป็นหัวข้อ เช่น คำจำกัดความ, อาการแสดงหลัก, และแนวทางการรักษา ให้ชัดเจน เข้าใจง่าย และเป็นภาษาไทยทางการแพทย์ที่สระสลวย
3. **CRITICAL LANGUAGE RULE**: You MUST output the ENTIRE response STRICTLY and EXCLUSIVELY in the THAI LANGUAGE (ภาษาไทย). Under NO circumstances are you allowed to output any Chinese (中文) or other languages. Think and write ONLY in Thai.
4. **NO CHINESE RULE**: ห้ามมีตัวอักษรจีน (Chinese Characters เช่น 丧失意识) ปรากฏออกมาเด็ดขาด หากต้องการอ้างอิงคำภาษาอังกฤษ ให้เขียนเป็นภาษาไทยคู่ภาษาอังกฤษธรรมดา เช่น หมดสติ (Loss of consciousness)
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
   - Diabetes -> เบาหวาน
   - Jaundice -> ดีซ่าน
   - Chicken Pox -> โรคอีสุกอีใส
   - Typhoid -> ไข้ไทฟอยด์
   - Hepatitis A -> โรคตับอักเสบ เอ
4. **CRITICAL LANGUAGE RULE**: You MUST output the ENTIRE response STRICTLY and EXCLUSIVELY in the THAI LANGUAGE (ภาษาไทย). Under NO circumstances are you allowed to output any Chinese (中文) or other languages. Think and write ONLY in Thai.
5. **NO CHINESE RULE**: ห้ามมีตัวอักษรจีน (Chinese Characters) ปรากฏออกมาเด็ดขาด หากต้องการอ้างอิงคำศัพท์ ให้แปลเป็นภาษาไทยสากลร่วมกับวงเล็บภาษาอังกฤษเท่านั้น
6. หากข้อมูลใน RAG มีอาการไม่ตรงกับผู้ป่วย ให้วิเคราะห์แย้งได้เลย (Critical Evaluation)
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