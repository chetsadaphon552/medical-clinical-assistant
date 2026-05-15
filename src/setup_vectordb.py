"""
Setup FAISS vector database with document chunking.
"""
import json
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', 'data/vectordb/faiss_index')


def chunk_text(text, chunk_size=400, overlap=80):
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk (default: 400)
        overlap: Number of characters to overlap between chunks (default: 80)
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not at the end, try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?)
            last_period = text[start:end].rfind('. ')
            last_exclaim = text[start:end].rfind('! ')
            last_question = text[start:end].rfind('? ')
            last_newline = text[start:end].rfind('\n')
            
            boundary = max(last_period, last_exclaim, last_question, last_newline)
            
            if boundary > chunk_size * 0.5:  # Only break if boundary is not too early
                end = start + boundary + 1
            else:
                # Fall back to word boundary
                last_space = text[start:end].rfind(' ')
                if last_space > chunk_size * 0.5:
                    end = start + last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text) or len(chunks) > 50:
            break
    
    return chunks


def setup_vectordb():
    """Setup FAISS vector database with chunked documents."""
    
    print("🔧 Setting up vector database...")
    
    # Load documents
    print("📚 Loading documents...")
    with open('data/processed/documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Chunk documents
    print("✂️  Chunking documents...")
    chunks = []
    
    for doc in documents:
        doc_chunks = chunk_text(doc['text'], chunk_size=400, overlap=80)
        
        for chunk_text_content in doc_chunks:
            chunks.append({
                'text': chunk_text_content,
                'metadata': doc['metadata']
            })
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"   Average: {len(chunks)/len(documents):.1f} chunks per document")
    
    # Load embedding model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings
    print("🔢 Generating embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    print("🔄 Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings.astype('float32'))
    
    # Create FAISS index with Inner Product (for normalized vectors = cosine similarity)
    print("📊 Creating FAISS index with cosine similarity...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save index and chunks
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    
    print("💾 Saving vector database...")
    faiss.write_index(index, os.path.join(VECTOR_DB_PATH, 'index.faiss'))
    
    with open(os.path.join(VECTOR_DB_PATH, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"✅ Vector database saved to: {VECTOR_DB_PATH}")
    print(f"   - Index: {len(chunks)} vectors of dimension {dimension}")
    print(f"   - Chunks: {len(chunks)} text chunks with metadata")
    print(f"   - Similarity: Cosine similarity (normalized vectors)")
    
    # Show statistics
    print("\n📊 Statistics:")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg chunks/doc: {len(chunks)/len(documents):.1f}")
    print(f"   Embedding dimension: {dimension}")
    
    # Show sample chunk
    print("\n📄 Sample chunk:")
    print("=" * 80)
    print(f"Condition: {chunks[0]['metadata']['condition']}")
    print(f"Text: {chunks[0]['text'][:300]}...")
    print("=" * 80)


if __name__ == '__main__':
    setup_vectordb()
