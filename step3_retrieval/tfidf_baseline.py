import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- load data ----------

DATA_PATH = "/Users/bhumitarora/Desktop/Auto_assessment_recommender/data/processed_assessments.csv"

df = pd.read_csv(DATA_PATH)

# Safety: drop empty text rows
df = df[df["rich_text"].notna()]
df = df[df["rich_text"].str.len() > 0]

documents = df["rich_text"].tolist()


# ---------- TF-IDF model ----------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # unigrams + bigrams
    max_df=0.9,
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(documents)


# ---------- retrieval function ----------

def retrieve_top_k(query, max_duration=None, k=5):
    """
    Retrieve top-k assessments, strictly filtering by max_duration if provided.
    """
    # 1. Filter by Duration FIRST (The "Math" Part)
    if max_duration:
        # Filter the DataFrame to only include valid times
        # We use copy() to avoid SettingWithCopy warnings
        filtered_df = df[df["duration_mins"] <= max_duration].copy()
    else:
        filtered_df = df.copy()

    # If filtering leaves no results, return empty
    if filtered_df.empty:
        return []

    # 2. Vectorize ONLY the filtered documents
    # Note: For TF-IDF, we usually fit on ALL data, but transform only the subset.
    # To keep it simple for this baseline, we can index into the main matrix.
    
    # Get indices of the filtered rows
    filtered_indices = filtered_df.index.tolist()
    
    # Transform query
    query_vec = vectorizer.transform([query])
    
    # Calculate similarity only for the filtered subset
    # We slice the main matrix using the indices
    subset_matrix = tfidf_matrix[filtered_indices]
    
    similarities = cosine_similarity(query_vec, subset_matrix)[0]
    
    # Get top indices (relative to the subset)
    top_relative_indices = similarities.argsort()[::-1][:k]
    
    results = []
    for rel_idx in top_relative_indices:
        original_idx = filtered_indices[rel_idx] # Map back to original DF
        results.append({
            "name": df.loc[original_idx, "name"],
            "url": df.loc[original_idx, "url"],
            "duration": df.loc[original_idx, "duration_mins"], # Return duration to verify
            "score": float(similarities[rel_idx])
        })
        
    return results

# ---------- test run ----------
if __name__ == "__main__":
    # Test Query 6: "Python... max duration of 60 minutes"
    query_text = "Python SQL Java Script mid-level"
    results = retrieve_top_k(query_text, max_duration=60, k=5)

    print(f"\nQuery: {query_text} (Max 60 mins)\n")
    for r in results:
        print(f"Name: {r['name']} | Time: {r['duration']}m | Score: {r['score']:.4f}")
