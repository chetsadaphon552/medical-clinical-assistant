"""
Tools for Medical Symptom Assistant Agent.
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import faiss
from langchain.tools import tool

# Load environment variables
load_dotenv()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', 'data/vectordb/vector_store')

# Global vector store (loaded once)
_index = None
_chunks = None
_model = None


def get_vectorstore():
    """Get or load the vector store."""
    global _index, _chunks, _model
    
    if _index is None:
        print("📚 Loading vector store...")
        
        # Load model
        _model = SentenceTransformer(EMBEDDING_MODEL)
        
        try:
            print("📚 Loading vector store...")
            
            # Load model
            _model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Load FAISS index
            _index = faiss.read_index(os.path.join(VECTOR_DB_PATH, 'index.faiss'))
            
            # Load chunks
            with open(os.path.join(VECTOR_DB_PATH, 'chunks.pkl'), 'rb') as f:
                _chunks = pickle.load(f)
            
            print("✅ Vector store loaded")
        except Exception as e:
            import traceback
            print(f"❌ Error loading vector store: {e}")
            print(traceback.format_exc())
            raise e
    
    return _index, _chunks, _model


@tool
def search_symptoms(symptoms: str, k: int = 5) -> str:
    """
    Search for medical conditions based on symptoms.
    This is a semantic search tool that finds conditions matching the described symptoms.
    
    Args:
        symptoms: Description of symptoms (e.g., "fever, cough, sore throat")
        k: Number of results to return (default: 5)
    
    Returns:
        String with possible conditions and information
    """
    
    index, chunks, model = get_vectorstore()
    
    # Improved query expansion for better context
    expanded_query = f"patient experiencing {symptoms} symptoms medical condition"
    
    # Embed query and normalize for cosine similarity
    query_embedding = model.encode([expanded_query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
    
    # Search with k*3 to get enough candidates for unique filtering
    similarities, indices = index.search(np.array([query_embedding]).astype('float32'), k * 3)
    
    if len(indices[0]) == 0:
        return "No matching conditions found for the described symptoms."
    
    # Pre-collect and sort results by score descending
    results = []
    for idx, similarity in zip(indices[0], similarities[0]):
        results.append({'idx': idx, 'score': float(similarity)})
    
    # Sort by score descending (highest similarity first)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    output = ""
    seen_conditions = set()
    result_count = 0
    RELEVANCE_THRESHOLD = 0.60
    
    for item in results:
        idx = item['idx']
        score = item['score']
        
        if score < RELEVANCE_THRESHOLD:
            continue
            
        chunk = chunks[idx]
        condition_name = chunk['metadata']['condition']
        
        if condition_name in seen_conditions:
            continue
        
        seen_conditions.add(condition_name)
        result_count += 1
        
        # Format for LLM
        output += f"Condition: {condition_name}\n"
        output += f"Confidence Score: {score:.2f}\n"
        output += f"Clinical Content: {chunk['text']}\n"
        output += f"Source: {chunk['metadata'].get('source', 'Unknown')}\n\n"

        if result_count >= k:
            break
        
        output += "\n"
    
    if result_count == 0:
        print(f"⚠️ [Tool] No conditions met the relevance threshold ({RELEVANCE_THRESHOLD})")
        return "No relevant clinical conditions found matching these symptoms in the database. The query may be non-clinical or outside the system's knowledge base."

    print(f"✅ [Tool] Found {result_count} relevant unique conditions")
    return output


@tool
def get_condition_details(condition_name: str) -> str:
    """
    Get detailed information about a specific medical condition.
    Use this tool when the user asks for details, explanation, knowledge, warning signs, 
    or general information about a specific disease.
    
    Args:
        condition_name: Name of the condition (e.g., "Common Cold", "Diabetes", "Pneumonia")
    
    Returns:
        Comprehensive detailed information about the condition including symptoms, treatment, and clinical considerations
    """
    print(f"🔍 [Tool] Getting comprehensive details for: {condition_name}")
    
    index, chunks, model = get_vectorstore()
    
    # Enhanced query to get better semantic matches
    enhanced_query = f"{condition_name} disease symptoms treatment clinical information"
    query_embedding = model.encode([enhanced_query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
    
    # Search with more candidates
    similarities, indices = index.search(np.array([query_embedding]).astype('float32'), 30)
    
    # Find all chunks related to this condition with fuzzy matching
    condition_chunks = []
    condition_name_lower = condition_name.lower()
    
    for idx, similarity in zip(indices[0], similarities[0]):
        chunk = chunks[idx]
        chunk_condition = chunk['metadata']['condition'].lower()
        
        # Fuzzy match: exact match or contains the condition name
        if (chunk_condition == condition_name_lower or 
            condition_name_lower in chunk_condition or 
            chunk_condition in condition_name_lower):
            condition_chunks.append({
                'chunk': chunk,
                'similarity': float(similarity)
            })
    
    if not condition_chunks:
        # Fallback: try to find by highest similarity
        print(f"⚠️ [Tool] Exact match not found, using semantic search fallback")
        for idx, similarity in zip(indices[0][:5], similarities[0][:5]):
            if similarity > 0.5:  # Only include relevant results
                condition_chunks.append({
                    'chunk': chunks[idx],
                    'similarity': float(similarity)
                })
    
    if not condition_chunks:
        return f"ไม่พบข้อมูลของโรค '{condition_name}' ในฐานข้อมูล กรุณาตรวจสอบชื่อโรคหรือลองใช้ชื่อภาษาอังกฤษ"
    
    # Sort by similarity (highest first)
    condition_chunks = sorted(condition_chunks, key=lambda x: x['similarity'], reverse=True)
    
    # Build comprehensive output
    main_condition = condition_chunks[0]['chunk']['metadata']['condition']
    output = f"=== รายละเอียดโรค: {main_condition} ===\n\n"
    
    # Combine all relevant chunks
    all_texts = []
    for item in condition_chunks:
        chunk = item['chunk']
        all_texts.append(chunk['text'])
    
    # Join all information
    full_text = "\n\n".join(all_texts)
    output += full_text
    
    # Add metadata
    output += f"\n\n--- Metadata ---\n"
    output += f"Condition: {main_condition}\n"
    output += f"Category: {condition_chunks[0]['chunk']['metadata'].get('category', 'N/A')}\n"
    output += f"Severity: {condition_chunks[0]['chunk']['metadata'].get('severity', 'N/A')}\n"
    output += f"Retrieved Chunks: {len(condition_chunks)}\n"
    
    print(f"✅ [Tool] Retrieved {len(condition_chunks)} chunks for '{main_condition}'")
    return output


# List of all tools (only 2 active tools)
ALL_TOOLS = [
    search_symptoms,
    get_condition_details
]
