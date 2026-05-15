"""
Prepare documents from Gretel AI dataset
"""
import pandas as pd
import json
import os

def prepare_gretel_documents():
    """Prepare documents from Gretel AI dataset."""
    
    print("=" * 80)
    print("📚 Preparing documents from Gretel AI dataset...")
    print("=" * 80)
    
    # Load dataset
    df = pd.read_csv('data/raw/gretel/symptom_to_diagnosis.csv')
    
    print(f"\n📊 Dataset: {len(df)} samples, {df['output_text'].nunique()} diseases")
    
    # Group by disease
    documents = []
    
    for disease in df['output_text'].unique():
        disease_df = df[df['output_text'] == disease]
        symptom_texts = disease_df['input_text'].tolist()
        
        # Create rich document with multiple symptom descriptions
        doc_text = f"""Condition: {disease}
Category: General
Severity: moderate

Real Patient Symptom Descriptions:

"""
        # Add all symptom descriptions (they're already good quality)
        for i, text in enumerate(symptom_texts, 1):
            doc_text += f"{i}. {text}\n\n"
        
        # Add summary
        doc_text += f"""
Summary: Patients with {disease} report various symptoms as described above.
Total cases: {len(symptom_texts)}
"""
        
        documents.append({
            'text': doc_text,
            'metadata': {
                'condition': disease,
                'category': 'General',
                'severity': 'moderate',
                'sample_count': len(symptom_texts),
                'source': 'gretel'
            }
        })
    
    # Save documents
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/documents.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created {len(documents)} documents")
    print(f"📁 Saved to: {output_file}")
    
    # Show all diseases
    print("\n📋 All diseases:")
    print("-" * 80)
    for i, doc in enumerate(sorted(documents, key=lambda x: x['metadata']['condition']), 1):
        print(f"{i:2d}. {doc['metadata']['condition']:35s} ({doc['metadata']['sample_count']} samples)")
    
    # Show sample
    print("\n📄 Sample document:")
    print("=" * 80)
    sample = documents[0]
    print(f"Condition: {sample['metadata']['condition']}")
    print(f"Samples: {sample['metadata']['sample_count']}")
    print(f"\nFirst 3 symptom descriptions:")
    lines = sample['text'].split('\n')
    count = 0
    for line in lines:
        if line.strip() and line[0].isdigit():
            print(f"  {line[:100]}...")
            count += 1
            if count >= 3:
                break
    print("=" * 80)
    
    return documents


if __name__ == '__main__':
    prepare_gretel_documents()
