"""
Test script to verify correct tool routing and tailored generative prompts.
"""
import sys
import os

# Add workspace directory to path
sys.path.append(os.getcwd())

from src.agent import MedicalSymptomAssistant

def test():
    print("🏥 Starting agent tests...")
    agent = MedicalSymptomAssistant()
    
    # Test 1: get_condition_details
    query_1 = "ขอรายละเอียดและข้อมูลเชิงลึกของโรคปอดอักเสบหน่อยครับ"
    print(f"\n💬 Test 1 Query: {query_1}")
    response_1 = agent.query(query_1)
    print("\n🏥 Response 1:")
    print(response_1)
    
    # Test 2: get_warning_signs
    query_2 = "สัญญาณอันตรายของโรคไข้เลือดออกที่ต้องรีบไปโรงพยาบาลทันทีมีอะไรบ้าง"
    print(f"\n💬 Test 2 Query: {query_2}")
    response_2 = agent.query(query_2)
    print("\n🏥 Response 2:")
    print(response_2)

if __name__ == "__main__":
    test()
