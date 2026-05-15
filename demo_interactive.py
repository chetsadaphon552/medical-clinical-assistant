"""
Interactive Demo - Medical Symptom Assistant
พิมพ์อาการเป็นภาษาไทยหรืออังกฤษได้เลย
"""
from src.agent import MedicalSymptomAssistant


def main():
    """Run interactive demo."""
    print("\n" + "=" * 80)
    print("🏥 ระบบสนับสนุนการตัดสินใจทางคลินิก / CLINICAL DECISION SUPPORT SYSTEM")
    print("=" * 80)
    print("⚠️  คำเตือน: ระบบนี้เป็นเครื่องมือช่วยวิเคราะห์ข้อมูลเบื้องต้นสำหรับบุคลากรทางการแพทย์")
    print("   CAUTION: This is a clinical decision support tool for healthcare professionals.")
    print("   Final diagnosis and treatment plans remain the responsibility of the physician.")
    print("=" * 80)
    print("\n💡 วิธีใช้ (Usage):")
    print("   - กรอกอาการแสดงของผู้ป่วย (Enter patient clinical presentation)")
    print("   - ระบบจะทำการสืบค้นข้อมูลและวิเคราะห์ Differential Diagnosis (DDx)")
    print("   - พิมพ์ 'quit', 'exit', 'ออก' หรือ 'จบ' เพื่อออกจากระบบ")
    print("=" * 80)
    
    # Initialize agent
    print("\n🔧 กำลังเริ่มระบบ (Initializing Clinical CDS Agent)...")
    agent = MedicalSymptomAssistant()
    
    print("\n✅ พร้อมใช้งาน (System Ready)!")
    print("=" * 80)
    
    # Interactive loop
    while True:
        try:
            print("\n" + "=" * 80)
            user_input = input("💬 อาการของคุณ / Your symptoms: ").strip()
            print("=" * 80)
            
            if not user_input:
                print("⚠️  กรุณาพิมพ์อาการของคุณ")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'ออก', 'จบ']:
                print("\n👋 ดูแลสุขภาพด้วยนะครับ / Take care!")
                print("   อย่าลืมปรึกษาแพทย์เมื่อมีข้อสงสัย")
                break
            
            # Get response
            print("\n⏳ กำลังวิเคราะห์อาการ...")
            try:
                response = agent.query(user_input)
                
                print("\n" + "=" * 80)
                print("🏥 คำตอบจากระบบ:")
                print("=" * 80)
                # Use encode/decode as a safeguard for terminal printing issues
                print(response.encode('utf-8', errors='replace').decode('utf-8'))
                print("=" * 80)
            except UnicodeDecodeError:
                print("\n❌ ข้อผิดพลาด: ระบบไม่สามารถอ่านตัวอักษรบางตัวได้ (Encoding Error)")
                print("   แนะนำให้ลองพิมพ์ใหม่ หรือเปลี่ยนไปรันผ่านคำสั่ง 'python demo_interactive.py' โดยตรงครับ")
            
        except KeyboardInterrupt:
            print("\n\n👋 ขอบคุณที่ใช้บริการ / Thank you!")
            break
        except Exception as e:
            # Print error safely
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"\n❌ เกิดข้อผิดพลาด: {error_msg}")
            print("   กรุณาลองใหม่อีกครั้ง")


if __name__ == '__main__':
    main()
