import logging
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from src.tools import search_symptoms, get_condition_details, get_warning_signs

# Load environment variables
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5-omni-7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Configure logging
logging.basicConfig(level=logging.INFO)
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
   - ใช้ภาษาไทยระดับทางการแพทย์ที่สละสลวย 100% (ห้ามมีภาษาอังกฤษปนในประโยคภาษาไทย ยกเว้นชื่อโรคในวงเล็บ)
   - **ห้าม** มีภาษาจีน เวียดนาม หรือภาษาอื่นๆ หลุดออกมาเด็ดขาด
   - แปลศัพท์เทคนิคให้เป็นภาษาไทยมาตรฐาน (เช่น Sputum -> เสมหะ, Rusty -> สีสนิมเหล็ก, Malaise -> ไม่สบายเนื้อไม่สบายตัว)

2. **การวิเคราะห์ (Clinical Analysis):**
   - วิเคราะห์เปรียบเทียบ (Differential Diagnosis) เพื่อหาจุดต่างของแต่ละโรค ห้ามเขียนทวนอาการผู้ป่วยซ้ำๆ
   - อ้างอิงข้อมูลจากฐานข้อมูล (RAG) เท่านั้น ห้ามแต่งข้อมูลเอง
   - เรียงลำดับโรคตามคะแนนความมั่นใจจาก **มากไปน้อย** เสมอ

3. **รูปแบบ (Formatting):**
   - ใช้ Markdown Table สำหรับรายชื่อโรค
   - เว้นบรรทัด (Double Newline) ระหว่างหัวข้อใหญ่เพื่อให้ดูสะอาดตา
   - ใช้หัวข้อหลักดังนี้: ### 📊 รายชื่อโรคที่เป็นไปได้, ### 🧠 บทวิเคราะห์ทางคลินิก, ### ⚠️ ข้อพิจารณาเพิ่มเติม

    *ตารางพจนานุกรมแปลศัพท์ (Internal Translation Map):*
    - Chest tightness -> แน่นหน้าอก
    - Shortness of breath / Difficulty breathing -> หายใจลำบาก / หอบเหนื่อย
    - Fatigue / Always tired -> อ่อนเพลีย / เหนื่อยล้าตลอดเวลา
    - Chills -> หนาวสั่น
    - Common Cold -> หวัดทั่วไป
    - Pneumonia -> ปอดอักเสบ
""")
        logger.info(f"✅ LLM initialized: {QWEN_MODEL} via API")
        
        # Available tools
        self.tools = {
            'search_symptoms': search_symptoms,
            'get_condition_details': get_condition_details,
            'get_warning_signs': get_warning_signs
        }
        logger.info(f"✅ Loaded {len(self.tools)} tools")

    def _select_tool(self, user_input: str) -> tuple:
        """Select appropriate tool based on user input using LLM reasoning."""
        
        prompt = f"""Analyze the user query and select the best medical tool.

**RULES:**
1. If the query is a greeting (hello, hi), goodbye (bye, farewell), thank you, or meta-comment about the AI (how do you work, why can you answer), you MUST select 'none'.
2. Select 'search_symptoms' ONLY if the user describes actual physical or mental symptoms.
3. Select 'none' if the query is vague or unrelated to a specific health concern.

User Query: {user_input}

Available Tools:
1. 'search_symptoms': For searching possible conditions based on symptoms.
2. 'get_condition_details': For detailed information about a specific known condition.
3. 'get_warning_signs': For warning signs or emergency indicators of a condition.
4. 'none': For greetings, goodbyes, general chat, or non-medical topics.

Output ONLY the tool name and its single most important argument in JSON format:
{{"tool": "tool_name", "argument": "value"}}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import json
            import re
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', response.content)
            if match:
                data = json.loads(match.group())
                tool = data.get('tool', 'search_symptoms')
                arg = data.get('argument', user_input)
                
                if tool == 'get_condition_details':
                    return 'get_condition_details', {'condition_name': arg}
                elif tool == 'get_warning_signs':
                    return 'get_warning_signs', {'condition_name': arg}
                elif tool == 'none':
                    return 'none', None
                else:
                    return 'search_symptoms', {'query': arg}
            return 'search_symptoms', {'query': user_input}
        except Exception as e:
            logger.error(f"Error selecting tool: {e}")
            return 'search_symptoms', {'query': user_input}

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
                trans_prompt = f"Translate the following Thai medical query to English. Output ONLY the translation: {user_input}"
                user_input = self.llm.invoke([HumanMessage(content=trans_prompt)]).content
                logger.info(f"📥 Translated Query: {user_input}")

            # Step 3: Tool selection
            logger.info("🤔 Analyzing symptoms...")
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
            
            # Use original input language for response
            prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
อาการของผู้ป่วย: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล:
{tool_result}

คำสั่ง: วิเคราะห์ข้อมูลและสร้างรายงานตามรูปแบบที่กำหนด (ภาษาไทย 100%)
กฎเหล็ก:
1. ห้ามมีภาษาเวียดนาม หรืออังกฤษปนในบทวิเคราะห์ (ยกเว้นชื่อโรคในวงเล็บ)
2. เรียงลำดับโรคตามคะแนนความมั่นใจจากมากไปน้อย
3. ใช้คำศัพท์ทางการแพทย์ไทยที่ถูกต้อง (เช่น เสมหะ แทน เสมหิ)
4. หากคะแนนใกล้เคียงกัน ให้เน้นการเปรียบเทียบจุดต่าง (Differential Diagnosis)
"""
            
            response = self.llm.invoke([self.system_message, HumanMessage(content=prompt)])
            
            logger.info("✅ Query processed successfully")
            return response.content
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Error processing query: {e}")
            logger.error(error_trace)
            return f"ขออภัยครับ เกิดข้อผิดพลาดในระบบ: {str(e)}"

# Singleton instance
assistant = MedicalSymptomAssistant()