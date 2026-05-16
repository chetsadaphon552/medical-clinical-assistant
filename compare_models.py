import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def chunk_text(text, chunk_size=400, overlap=80):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = max(text[start:end].rfind('. '), text[start:end].rfind('! '), 
                           text[start:end].rfind('? '), text[start:end].rfind('\n'))
            if boundary > chunk_size * 0.5:
                end = start + boundary + 1
            else:
                last_space = text[start:end].rfind(' ')
                if last_space > chunk_size * 0.5:
                    end = start + last_space
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start = end - overlap
        if start >= len(text) or len(chunks) > 50: break
    return chunks

def get_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    # 1. Models to compare
    model_names = [
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    print("📚 Loading and chunking clinical database...")
    with open('data/processed/documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
        
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc['text'])
        for text in doc_chunks:
            chunks.append({'text': text, 'metadata': doc['metadata']})
    print(f"✅ Created {len(chunks)} chunks.")

    # 2. Load models and encode chunks
    models = {}
    encoded_chunks = {}
    
    for name in model_names:
        print(f"\n🤖 Loading Model: {name} ...")
        models[name] = SentenceTransformer(name)
        print(f"🔢 Encoding database with {name} ...")
        texts = [c['text'] for c in chunks]
        encoded_chunks[name] = models[name].encode(texts, show_progress_bar=False)

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    llm = ChatOpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model=os.getenv("QWEN_MODEL", "qwen2.5-omni-7b"),
        temperature=0.01,
    )

    print("\n" + "="*60)
    print("🧪 READY FOR EXPERIMENT (Full Pipeline Mirror)")
    print("="*60)
    
    # 3. Interactive testing loop
    while True:
        print("\n" + "-"*60)
        query = input("พิมพ์อาการที่ต้องการทดสอบ (ภาษาไทย) หรือ 'q' เพื่อออก: ")
        if query.lower() == 'q': break
        if not query.strip(): continue
        
        # Step A: Translate Thai to English (Mirroring real app)
        print("🌐 [Qwen] Translating Thai to English...")
        trans_prompt = f"Translate the following Thai text to English. Output ONLY the exact translation without any explanation: {query}"
        translated_query = llm.invoke([HumanMessage(content=trans_prompt)]).content
        print(f"📥 Translated: {translated_query}")
        
        # Expand query similar to real system
        expanded_query = f"patient experiencing {translated_query} symptoms medical condition"
        
        for name in model_names:
            print(f"\n📊 Model: {name}")
            query_vec = models[name].encode([expanded_query])[0]
            
            # Calculate similarities
            results = []
            for i, chunk_vec in enumerate(encoded_chunks[name]):
                sim = get_cosine_similarity(query_vec, chunk_vec)
                results.append({'score': sim, 'condition': chunks[i]['metadata']['condition']})
            
            # Sort by highest similarity
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Filter unique conditions for Top 3
            seen = set()
            unique_top = []
            for r in results:
                if r['condition'] not in seen:
                    seen.add(r['condition'])
                    unique_top.append(r)
                if len(unique_top) == 3: break
                
            for i, res in enumerate(unique_top):
                print(f"   {i+1}. {res['condition'].title()} (Score: {res['score']:.4f})")

if __name__ == "__main__":
    main()
