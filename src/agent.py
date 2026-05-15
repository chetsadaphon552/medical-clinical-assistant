"""
Medical Clinical Assistant Agent - Provides clinical decision support for healthcare professionals.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.tools import search_symptoms, get_condition_details, filter_by_severity, get_warning_signs
from src.logger import setup_logger

# Load environment variables
load_dotenv()

QWEN_API_KEY = os.getenv('QWEN_API_KEY')
QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen2.5-omni-7b')
QWEN_BASE_URL = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

# Setup logger
logger = setup_logger()


class MedicalSymptomAssistant:
    """
    AI Clinical Decision Support Assistant - Assists healthcare professionals with differential diagnosis.
    
    IMPORTANT: This is a decision support tool intended for use by qualified healthcare professionals.
    Final clinical judgment and diagnosis remain the responsibility of the treating physician.
    """
    
    def __init__(self):
        """Initialize the assistant."""
        logger.info("🏥 Initializing Clinical Decision Support Assistant...")
        
        # Initialize LLM with multilingual support via DashScope API
        self.llm = ChatOpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model=QWEN_MODEL,
            temperature=0.01,  # Strict adherence to instructions
            max_tokens=1000,   # Increased to prevent Thai character truncation
        )
        
        self.system_message = SystemMessage(content="""คุณคือ "Clinical Decision Support Assistant" ผู้ช่วยอัจฉริยะที่ออกแบบมาเพื่อสนับสนุนบุคลากรทางการแพทย์ในการวินิจฉัยแยกโรค (Differential Diagnosis - DDx) โดยใช้ฐานข้อมูลทางคลินิกอ้างอิง

แนวทางปฏิบัติในการตอบคำถาม (Operational Protocols):

1. แหล่งข้อมูล (Data Grounding): 
   - คุณต้องวิเคราะห์และตอบคำถามโดยใช้ข้อมูลจาก "เครื่องมือค้นหา (RAG)" เท่านั้น 
   - ห้ามแต่งเติมหรือใช้ความรู้ภายนอกที่ไม่ปรากฏในข้อมูลที่ดึงมา (No Hallucination)
   - หากข้อมูลที่ดึงมาไม่สอดคล้องกับอาการของผู้ป่วย ให้ระบุว่า "ไม่พบรูปแบบอาการที่ตรงกับฐานข้อมูลทางคลินิก"

2. การจัดการคำถามที่ไม่เกี่ยวข้อง (Out-of-Scope Handling):
   - หากคำถามไม่เกี่ยวกับการเจ็บป่วยหรือสุขภาพ (เช่น เรื่องทั่วไป, การ์ตูน, สภาพอากาศ) ให้ตอบว่า: "ขออภัยครับ ผมได้รับการออกแบบมาเพื่อทำหน้าที่เป็นผู้ช่วยสนับสนุนการตัดสินใจทางคลินิกเท่านั้น จึงไม่สามารถให้ข้อมูลในหัวข้ออื่นได้ กรุณาระบุอาการทางคลินิกที่ต้องการวิเคราะห์ครับ"

3. โครงสร้างรายงานที่ต้องทำตามเป๊ะๆ (Strict Formatting Rules):
   - ทุกอันดับโรคต้อง "ขึ้นบรรทัดใหม่" เสมอ
   - ใช้หัวข้อ Markdown (###) ตามที่กำหนด
   - ห้ามใช้ภาษาอื่นนอกจากไทยและอังกฤษ (ห้ามมีภาษาจีน หรือเวียดนาม เช่น xét nghiệm เด็ดขาด)
   - ใช้ Bullet points (-) ในส่วนของข้อพิจารณาเพิ่มเติม

Template ตัวอย่าง (Strict Structure):

   ### 📊 รายชื่อโรคที่เป็นไปได้ (Possible Conditions)

   | ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
   | :--- | :--- | :--- |
   | 1 | [ชื่อโรคภาษาไทย (English Name)] | [0.xx] |
   | 2 | [ชื่อโรคภาษาไทย (English Name)] | [0.xx] |

   ### 🧠 บทวิเคราะห์ทางคลินิก (Clinical Analysis)
   [บทวิเคราะห์โดยละเอียดที่เชื่อมโยงกับอาการของผู้ป่วย]

   ### ⚠️ ข้อพิจารณาเพิ่มเติม (Additional Considerations)
   - **ปัจจัยเสี่ยง:** [รายละเอียด]
   - **การสืบค้นแนะนำ:** [รายละเอียด]

   ---
   *คำเตือน: ข้อมูลนี้ใช้เพื่อประกอบการตัดสินใจทางคลินิกเท่านั้น การวินิจฉัยขั้นสุดท้ายขึ้นอยู่กับดุลยพินิจของแพทย์ผู้ตรวจ*

4. ภาษาและสำนวน (Language Rules - STRICT):
   - **ห้ามมีคำภาษาอังกฤษปนในบทวิเคราะห์เด็ดขาด** (ยกเว้นชื่อโรคในวงเล็บ): เช่น ห้ามเขียนว่า "รู้สึก sore throat" ให้แปลเป็น "เจ็บคอ" ทันที
   - **แปลหลักฐานทุกอย่างจาก RAG เป็นภาษาไทย**: ข้อมูลที่ดึงมาเป็นภาษาอังกฤษ คุณต้องแปลเป็นภาษาไทยที่อ่านง่ายและลื่นไหล 100%
   - **ตารางต้องเป๊ะ**: ต้องใช้เครื่องหมาย `|` และ `-` ให้ครบตามรูปแบบ Markdown Table เพื่อให้ระบบแสดงผลเป็นตารางจริงๆ

   *พจนานุกรมช่วยแปล:*
   - sore throat -> เจ็บคอ, coughing a lot -> ไอมาก, chills -> หนาวสั่น, run down/weak -> อ่อนเพลีย, fever -> มีไข้, chest tightness -> แน่นหน้าอก, always tired -> เหนื่อยตลอดเวลา
""")
        logger.info(f"✅ LLM initialized: {QWEN_MODEL} via API")
        
        # Available tools
        self.tools = {
            'search_symptoms': search_symptoms,
            'get_condition_details': get_condition_details,
            'filter_by_severity': filter_by_severity,
            'get_warning_signs': get_warning_signs
        }
        logger.info(f"✅ Loaded {len(self.tools)} tools")
    
    def _select_tool(self, user_input: str) -> tuple:
        """Select appropriate tool based on user input using LLM reasoning."""
        
        prompt = f"""Analyze the user query and select the best medical tool.
If the query is NOT RELATED to medical symptoms, health conditions, or clinical advice, select 'none'.

User Query: {user_input}

Available Tools:
1. 'search_symptoms': For searching possible conditions based on symptoms.
2. 'get_condition_details': For detailed information about a specific known condition.
3. 'get_warning_signs': For warning signs or emergency indicators of a condition.
4. 'none': Use this ONLY if the query is unrelated to medicine or health.

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
                    return 'none', {}
            
            return 'search_symptoms', {'symptoms': user_input, 'k': 5}
        except:
            return 'search_symptoms', {'symptoms': user_input, 'k': 5}
    
    def _translate_to_english(self, thai_text: str) -> str:
        """Translate Thai symptoms to English for better RAG search."""
        logger.info("🌐 Translating Thai to English for better search...")
        
        # Enhanced fallback dictionary for common symptoms
        thai_to_english = {
            # Symptoms
            'ไข้': 'fever',
            'ไอ': 'cough',
            'เจ็บคอ': 'sore throat',
            'ปวดหัว': 'headache',
            'ปวดท้อง': 'stomach pain',
            'คลื่นไส้': 'nausea',
            'อาเจียน': 'vomiting',
            'ท้องเสีย': 'diarrhea',
            'ท้องผูก': 'constipation',
            'คัน': 'itchy',
            'ผื่น': 'rash',
            'แสงสว่าง': 'light',
            'เจ็บตา': 'eye pain',
            'ตาแดง': 'red eyes',
            'น้ำมูก': 'runny nose',
            'จาม': 'sneezing',
            'หายใจลำบาก': 'difficulty breathing',
            'เหนื่อย': 'fatigue',
            'อ่อนเพลีย': 'weakness',
            'ง่วง': 'drowsy',
            'นอนไม่หลับ': 'insomnia',
            'เวียนหัว': 'dizziness',
            'ปวดเมื่อย': 'body ache',
            'ปวดกล้ามเนื้อ': 'muscle pain',
            'ปวดข้อ': 'joint pain',
            'บวม': 'swelling',
            'ผิวหนัง': 'skin',
            'คอแห้ง': 'dry throat',
            'เสมหะ': 'phlegm',
            'หนาว': 'chills',
            'สั่น': 'shivering',
            'เลือดออก': 'bleeding',
            
            # Severity
            'มาก': 'severe',
            'มากๆ': 'very severe',
            'น้อย': 'mild',
            'เล็กน้อย': 'slight',
            'อ่อน': 'mild',
            'รุนแรง': 'severe',
            
            # Location
            'ข้างซ้าย': 'left side',
            'ข้างขวา': 'right side',
            'ด้านหน้า': 'front',
            'ด้านหลัง': 'back',
            
            # Connectors
            'และ': 'and',
            'มี': 'have',
            'รู้สึก': 'feel',
            'เป็น': 'have'
        }
        
        prompt = f"""You are an expert medical translator. Your task is to accurately translate the following Thai patient symptoms into English.
These English symptoms will be used for a semantic search in a medical database containing patient symptom descriptions.

Guidelines:
1. Translate accurately into common English medical symptom terms (e.g., "ไข้สูง" -> "high fever", "คลื่นไส้" -> "nausea", "ปวดเมื่อย" -> "body aches").
2. Maintain the severity and specific locations if mentioned (e.g., "ปวดหัวข้างซ้ายมาก" -> "severe left-sided headache").
3. DO NOT include any conversational filler in your output (do not write "The patient has", just output the symptoms).
4. ONLY output the translated English text, nothing else.

Thai symptoms: {thai_text}

English translation:"""
        
        try:
            messages = [self.system_message, HumanMessage(content=prompt)]
            english_text = self.llm.invoke(messages).content.strip()
            logger.info(f"✅ Translated: {english_text}")
            return english_text
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            logger.info("🔄 Using fallback dictionary translation...")
            
            # Fallback: simple word replacement
            english_text = thai_text.lower()
            for thai, english in thai_to_english.items():
                english_text = english_text.replace(thai, f' {english} ')
            
            # Clean up extra spaces
            english_text = ' '.join(english_text.split())
            
            logger.info(f"✅ Fallback translation: {english_text}")
            return english_text
    
    def query(self, user_input: str) -> str:
        """
        Process user query and return health information.
        
        Args:
            user_input: User's symptom description
        
        Returns:
            Assistant's response with health information
        """
        logger.info("=" * 80)
        logger.info(f"👤 User Query: {user_input}")
        logger.info("=" * 80)
        
        try:
            # Detect language
            is_thai = any('\u0E00' <= char <= '\u0E7F' for char in user_input)
            original_input = user_input
            
            # Step 1: Translate Thai to English for better RAG search
            if is_thai:
                user_input = self._translate_to_english(user_input)
            
            # Step 2: Analyze query and select tool
            logger.info("🤔 Analyzing symptoms...")
            
            tool_name, tool_input = self._select_tool(user_input)
            
            if tool_name == 'none':
                logger.info("🚫 Query identified as non-clinical.")
                return "ขออภัยครับ ระบบนี้เป็นระบบสนับสนุนการตัดสินใจทางคลินิก (Clinical CDS) ซึ่งออกแบบมาเพื่อวิเคราะห์อาการเจ็บป่วยและข้อมูลทางการแพทย์เท่านั้น กรุณาระบุอาการแสดงของผู้ป่วยเพื่อเริ่มการวิเคราะห์ครับ"

            logger.info(f"🔧 Selected tool: {tool_name}")
            logger.info(f"📥 Tool input: {tool_input}")
            
            # Step 3: Call the tool
            logger.info("⚙️  Executing tool (RAG search)...")
            tool = self.tools[tool_name]
            tool_result = tool.invoke(tool_input)
            
            logger.info("✅ Tool execution complete")
            logger.info(f"📊 Retrieved information:")
            logger.info(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)
            
            # Step 4: Generate natural language response
            logger.info("💭 Generating response with LLM...")
            
            # Use original input language for response
            if is_thai:
                prompt = f"""คุณคือผู้ช่วยตัดสินใจทางคลินิก (Clinical Decision Support Assistant)
อาการของผู้ป่วย: "{original_input}"

ข้อมูลอ้างอิงจากฐานข้อมูล:
{tool_result}

คำสั่ง: วิเคราะห์ข้อมูลและสร้างรายงานตามรูปแบบที่กำหนด (ภาษาไทย 100%)
**กฎเหล็ก: แสดงเพียง 3 อันดับแรกและเรียงตามคะแนนความมั่นใจจากมากไปน้อย**

### 📊 รายชื่อโรคที่เป็นไปได้ (Possible Conditions - Top 3)
| ลำดับ | รายชื่อโรค | คะแนนความมั่นใจ |
| :--- | :--- | :--- |
| 1 | [ชื่อไทย (English Name)] | [0.xx] |
| 2 | [ชื่อไทย (English Name)] | [0.xx] |
| 3 | [ชื่อไทย (English Name)] | [0.xx] |

### 🧠 บทวิเคราะห์ทางคลินิก (Clinical Analysis)
(วิเคราะห์อาการโดยแปลเป็นไทย 100% ห้ามมีภาษาอังกฤษปนในประโยค)

### ⚠️ ข้อพิจารณาเพิ่มเติม (Additional Considerations)
- **ปัจจัยเสี่ยง:** [รายละเอียด]
- **การสืบค้นแนะนำ:** [รายละเอียด]

---
**พจนานุกรมแปลโรคที่ต้องใช้ห้ามเพี้ยน:**
- Pneumonia -> ปอดอักเสบ
- Common Cold -> หวัดทั่วไป
- Dengue -> ไข้เลือดออก
- Influenza -> ไข้หวัดใหญ่
- Bronchial Asthma -> หอบหืด
- Allergy -> ภูมิแพ้
- Typhoid -> ไทฟอยด์
- Malaria -> มาลาเรีย

คำตอบ (ภาษาไทย 100%):"""
            else:
                prompt = f"""You are a Clinical Decision Support Assistant. Respond in English only.

Patient Presentation: "{user_input}"

Clinical Database Evidence
{tool_result}

Instructions: Analyze the evidence and formulate a Differential Diagnosis (DDx) report for a clinician using the structure below.

Report Structure:
1. Relevant Differential Diagnoses (DDx) (Show 1-3 items as found in database):
   - (Condition Name) [Confidence Score: X.XX]
   (Note: Use ONLY data from the clinical database. Do NOT hallucinate disease names or scores. If fewer than 3 relevant conditions are found, list only those. If no relevant conditions are found, state "No relevant data found".)

2. Clinical Evidence Correlation: (Briefly correlate the reported symptoms with the pathophysiology of the suggested conditions)

3. Follow-up Recommendations:
   - [Warning signs or complications to monitor]
   - [Suggested diagnostic investigations if relevant to the presentation]

4. Disclaimer: "This information is for clinical decision support only. Final diagnosis and management remain the responsibility of the treating physician."

Important:
- Use professional clinical terminology.
- Maintain an analytical and objective tone.

Response:"""
            
            try:
                messages = [self.system_message, HumanMessage(content=prompt)]
                response = self.llm.invoke(messages).content
            except Exception as llm_error:
                logger.error(f"❌ LLM failed: {llm_error}")
                logger.info("🔄 Using fallback response (direct RAG results)...")
                
                # Extract top 3 conditions from tool_result
                lines = tool_result.split('\n')
                top_3_lines = []
                condition_count = 0
                
                for line in lines:
                    if line.strip() and (line[0].isdigit() and '. ' in line[:5]):
                        condition_count += 1
                        if condition_count > 3:
                            break
                    if condition_count > 0 and condition_count <= 3:
                        top_3_lines.append(line)
                
                top_3_result = '\n'.join(top_3_lines)
                
                # Fallback: return top 3 RAG results
                if is_thai:
                    response = f"""ผลการวิเคราะห์ข้อมูลทางคลินิกเบื้องต้นสำหรับอาการ: "{original_input}"

วินิจฉัยแยกโรค (DDx) 3 อันดับแรกจากฐานข้อมูล:

{top_3_result}

⚠️ หมายเหตุ: ระบบประมวลผล LLM ไม่สำเร็จ ข้อมูลข้างต้นเป็นข้อมูลดิบจากฐานข้อมูลทางสถิติ
การวินิจฉัยขั้นสุดท้ายควรพิจารณาร่วมกับประวัติทางคลินิกและการตรวจร่างกายโดยละเอียด

ข้อพิจารณาเบื้องต้น:
- เฝ้าระวังอาการแสดงที่รุนแรง
- พิจารณาการส่งตรวจเพิ่มเติมตามแนวทางเวชปฏิบัติ"""
                else:
                    response = f"""Preliminary clinical analysis for presentation: "{user_input}"

Top 3 Differential Diagnoses (DDx) from evidence base:

{top_3_result}

⚠️ Note: LLM processing failed. Above is raw statistical evidence from the database.
Final diagnosis should be based on comprehensive clinical history and physical examination.

Clinical Considerations:
- Monitor for red flag symptoms
- Consider further investigations per clinical guidelines"""
            
            logger.info("=" * 80)
            logger.info("✅ Response Generated")
            logger.info("=" * 80)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return f"I apologize, but I encountered an error. Please try rephrasing your symptoms or consult a healthcare professional."
    
    def chat(self):
        """Start interactive chat session."""
        print("\n" + "=" * 80)
        print("🏥 ระบบสนับสนุนการตัดสินใจทางคลินิก / CLINICAL DECISION SUPPORT SYSTEM")
        print("=" * 80)
        print("⚠️  คำเตือน: ระบบนี้เป็นเครื่องมือช่วยวิเคราะห์ข้อมูลเบื้องต้นสำหรับบุคลากรทางการแพทย์")
        print("   CAUTION: This is a clinical decision support tool for healthcare professionals.")
        print("   Final diagnosis and treatment plans remain the responsibility of the physician.")
        print("=" * 80)
        print("\nกรอกอาการแสดงของผู้ป่วย (ภาษาไทยหรืออังกฤษ)")
        print("Enter patient clinical presentation (Thai or English)")
        print("พิมพ์ 'quit' หรือ 'exit' เพื่อออก / Type 'quit' or 'exit' to end")
        print("=" * 80 + "\n")
        
        while True:
            try:
                user_input = input("\n💬 อาการของคุณ / Your symptoms: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'ออก', 'จบ']:
                    print("\n👋 ดูแลสุขภาพด้วยนะครับ / Take care of your health! Remember to consult healthcare professionals when needed.")
                    break
                
                # Get response
                response = self.query(user_input)
                
                print(f"\n🏥 Clinical Assistant:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Stay healthy!")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                print(f"\n❌ Error: {e}")


def main():
    """Main function to run the assistant."""
    assistant = MedicalSymptomAssistant()
    assistant.chat()


if __name__ == '__main__':
    main()