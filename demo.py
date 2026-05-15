"""
Demo script for Medical Symptom Assistant.
Shows the complete Agentic RAG workflow.
"""
from src.agent import MedicalSymptomAssistant

def demo():
    """Run demo with sample queries."""
    
    print("\n" + "=" * 80)
    print("🎬 MEDICAL SYMPTOM ASSISTANT - DEMO")
    print("=" * 80)
    print("This demo shows the Agentic RAG workflow:")
    print("1. User describes symptoms")
    print("2. Agent analyzes and selects appropriate tool")
    print("3. RAG system searches medical knowledge base")
    print("4. LLM generates natural, empathetic response")
    print("=" * 80 + "\n")
    
    # Initialize assistant
    assistant = MedicalSymptomAssistant()
    
    # Sample queries
    queries = [
        "I have fever, cough, and sore throat",
        "I'm experiencing severe headache and sensitivity to light",
        "What are the warning signs for flu?",
        "I have stomach pain, nausea, and diarrhea"
    ]
    
    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"DEMO QUERY {i}/{len(queries)}")
        print("=" * 80)
        print(f"💬 User: {query}")
        print("-" * 80)
        
        response = assistant.query(query)
        
        print(f"\n🏥 Assistant: {response}")
        
        if i < len(queries):
            input("\n[Press Enter for next query...]")
    
    print("\n" + "=" * 80)
    print("🎬 DEMO COMPLETE")
    print("=" * 80)
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Semantic search (RAG)")
    print("   ✅ Agent tool selection")
    print("   ✅ Medical knowledge retrieval")
    print("   ✅ Natural language generation")
    print("   ✅ Observability (check logs/agent.log)")
    print("\n📊 To see detailed logs:")
    print("   cat logs/agent.log")
    print("\n🚀 To run interactive mode:")
    print("   python demo_interactive.py")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    demo()
