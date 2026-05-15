"""
Load Gretel AI Symptom to Diagnosis dataset
"""
from datasets import load_dataset
import json
import os
import pandas as pd

def load_gretel_dataset():
    """Load and explore Gretel AI dataset."""
    
    print("=" * 80)
    print("📥 Loading Gretel AI Symptom to Diagnosis dataset...")
    print("=" * 80)
    
    try:
        # Load dataset
        print("\n⏳ Downloading dataset from Hugging Face...")
        ds = load_dataset("gretelai/symptom_to_diagnosis")
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   Splits: {list(ds.keys())}")
        
        # Get train split
        train_data = ds['train']
        
        print(f"\n📈 Train split:")
        print(f"   Total samples: {len(train_data)}")
        print(f"   Columns: {train_data.column_names}")
        
        # Convert to pandas for easier exploration
        df = train_data.to_pandas()
        
        print(f"\n📄 Sample data:")
        print("-" * 80)
        print(df.head(3))
        
        # Check unique diagnoses
        if 'diagnosis' in df.columns:
            print(f"\n🏥 Unique diagnoses: {df['diagnosis'].nunique()}")
            print("\n📋 Top 10 diagnoses:")
            print(df['diagnosis'].value_counts().head(10))
        elif 'label' in df.columns:
            print(f"\n🏥 Unique labels: {df['label'].nunique()}")
            print("\n📋 Top 10 labels:")
            print(df['label'].value_counts().head(10))
        
        # Save to CSV for inspection
        output_dir = 'data/raw/gretel'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'symptom_to_diagnosis.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n💾 Saved to: {output_file}")
        
        # Show sample
        print("\n📄 Sample record:")
        print("=" * 80)
        sample = df.iloc[0]
        for col in df.columns:
            print(f"{col}: {sample[col][:200] if isinstance(sample[col], str) and len(str(sample[col])) > 200 else sample[col]}")
        print("=" * 80)
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n💡 Make sure you have 'datasets' library installed:")
        print("   pip install datasets")
        return None


if __name__ == '__main__':
    load_gretel_dataset()
