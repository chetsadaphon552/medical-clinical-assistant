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
        output += f"Clinical Content: {chunk['content']}\n"
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
    
    Args:
        condition_name: Name of the condition (e.g., "Common Cold", "Influenza")
    
    Returns:
        Detailed information about the condition
    """
    print(f"🔍 [Tool] Getting details for: {condition_name}")
    
    index, chunks, model = get_vectorstore()
    
    # Search for the condition
    query_embedding = model.encode([condition_name])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), 10)
    
    # Find chunks for this condition
    condition_chunks = []
    for idx in indices[0]:
        chunk = chunks[idx]
        if chunk['metadata']['condition'].lower() == condition_name.lower():
            condition_chunks.append(chunk)
    
    if not condition_chunks:
        return f"Condition '{condition_name}' not found in database."
    
    # Combine all chunks for this condition
    output = f"Detailed Information: {condition_chunks[0]['metadata']['condition']}\n"
    output += "=" * 80 + "\n\n"
    
    # Get full text from all chunks
    full_text = "\n".join([chunk['text'] for chunk in condition_chunks])
    output += full_text
    
    print(f"✅ [Tool] Retrieved details for {condition_chunks[0]['metadata']['condition']}")
    return output


@tool
def filter_by_severity(severity: str, symptoms: str = None) -> str:
    """
    Filter conditions by severity level.
    
    Args:
        severity: Severity level (mild, moderate, severe)
        symptoms: Optional symptoms to match
    
    Returns:
        String with filtered conditions
    """
    print(f"🔍 [Tool] Filtering by severity: {severity}")
    
    index, chunks, model = get_vectorstore()
    
    # Build query
    query = f"{severity} severity"
    if symptoms:
        query += f" {symptoms}"
    
    # Search
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), 15)
    
    # Filter by severity
    filtered = []
    seen_conditions = set()
    
    for idx in indices[0]:
        chunk = chunks[idx]
        condition_name = chunk['metadata']['condition']
        
        if (chunk['metadata']['severity'].lower() == severity.lower() and 
            condition_name not in seen_conditions):
            filtered.append(chunk)
            seen_conditions.add(condition_name)
    
    if not filtered:
        return f"No conditions found with {severity} severity."
    
    output = f"Conditions with {severity} severity:\n\n"
    
    for i, chunk in enumerate(filtered[:5], 1):
        output += f"{i}. {chunk['metadata']['condition']}\n"
        output += f"   Category: {chunk['metadata']['category']}\n"
        output += f"   Common symptoms: {', '.join(chunk['metadata']['symptoms'][:5])}\n\n"
    
    print(f"✅ [Tool] Found {len(filtered)} conditions")
    return output


@tool
def get_warning_signs(condition_name: str) -> str:
    """
    Get warning signs for when to seek immediate medical attention.
    
    Args:
        condition_name: Name of the condition
    
    Returns:
        Warning signs and when to see a doctor
    """
    print(f"🔍 [Tool] Getting warning signs for: {condition_name}")
    
    index, chunks, model = get_vectorstore()
    
    # Search for the condition
    query = f"{condition_name} warning signs seek medical attention"
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), 5)
    
    # Find relevant chunk
    for idx in indices[0]:
        chunk = chunks[idx]
        if condition_name.lower() in chunk['metadata']['condition'].lower():
            # Extract warning signs section
            text = chunk['text']
            if 'Warning Signs' in text or 'Seek Medical Attention' in text:
                lines = text.split('\n')
                output = f"⚠️  Warning Signs for {chunk['metadata']['condition']}:\n\n"
                
                in_warning_section = False
                for line in lines:
                    if 'Warning Signs' in line or 'Seek Medical Attention' in line:
                        in_warning_section = True
                        continue
                    if in_warning_section and line.strip():
                        output += f"{line}\n"
                
                output += "\n⚠️  If you experience any of these warning signs, seek medical attention immediately."
                
                print(f"✅ [Tool] Retrieved warning signs")
                return output
    
    return f"Warning signs for '{condition_name}' not found. Please consult a healthcare provider if symptoms worsen."


# List of all tools
ALL_TOOLS = [
    search_symptoms,
    get_condition_details,
    filter_by_severity,
    get_warning_signs
]
