import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Load environment variables from .env file
load_dotenv()

# Get the key securely
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Error: GEMINI_API_KEY not found in .env file")

INPUT_CSV = "/Users/bhumitarora/Desktop/Auto_assessment_recommender/data/processed_assessments.csv"
EMBEDDING_FILE = "/Users/bhumitarora/Desktop/Auto_assessment_recommender/data/assessment_embeddings_google.npy"

# Initialize LangChain's Google Embeddings wrapper
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY,
    task_type="retrieval_document"
)

def main():
    # 1. Load Data
    print("📂 Loading data...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}")
        return

    df = df[df["rich_text"].notna()]
    texts = df["rich_text"].tolist()
    
    print(f"⚡ Generating embeddings for {len(texts)} assessments using LangChain...")
    
    # 2. Generate Embeddings in Batches
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"   Processing batch {i} to {i+len(batch)}...")
        
        try:
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            time.sleep(1) # Respect rate limits
            
        except Exception as e:
            print(f"❌ Error on batch {i}: {e}")
            # Fill with zeros if failure
            all_embeddings.extend([[0]*768] * len(batch))

    # 3. Save
    embeddings_array = np.array(all_embeddings)
    print(f"💾 Saving embeddings to {EMBEDDING_FILE}...")
    np.save(EMBEDDING_FILE, embeddings_array)
    print("✅ Done! LangChain Embeddings saved.")

if __name__ == "__main__":
    main()