"""
Step 4b: Evaluation with LLM Reranking

Compares recall metrics:
  - Without LLM reranking (baseline hybrid search)
  - With LLM reranking (Gemini 1.5 Flash via LangChain)

Usage:
    python evaluate_with_llm_reranking.py
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# API endpoint - use local for testing
API_URL = "http://localhost:8000"


def normalize_url(url: str) -> str:
    """Extract slug from URL for matching."""
    return str(url).strip().lower().rstrip("/").split("/")[-1]


def get_ground_truth_slugs(query: str, df_truth: pd.DataFrame) -> set:
    """Get ground truth URL slugs for a query."""
    urls = df_truth[df_truth['Query'] == query]['Assessment_url'].tolist()
    return set(normalize_url(url) for url in urls)


def get_predictions(query: str, use_reranking: bool = False, k: int = 20) -> list:
    """Call API to get predictions."""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"query": query, "use_reranking": use_reranking},
            timeout=60  # LLM reranking may take longer
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("recommended_assessments", [])[:k]
            return [normalize_url(r.get("url", "")) for r in results]
    except Exception as e:
        print(f"  ⚠️ API Error: {e}")
    return []


def calculate_recall_at_k(pred_slugs: list, truth_slugs: set, k: int) -> float:
    """Calculate recall at K."""
    if not truth_slugs:
        return 0.0
    preds_at_k = set(pred_slugs[:k])
    hits = len(preds_at_k & truth_slugs)
    return hits / len(truth_slugs)


def evaluate(df_train: pd.DataFrame, use_reranking: bool = False, 
             k_values: list = [1, 5, 10, 20]) -> dict:
    """
    Evaluate recall at different K values.
    """
    recalls = {k: [] for k in k_values}
    query_results = []
    
    queries = df_train['Query'].unique()
    
    mode = "WITH LLM Reranking" if use_reranking else "WITHOUT LLM Reranking"
    print(f"\n📊 Evaluating {mode}...")
    print("-" * 50)
    
    for i, query in enumerate(queries):
        truth_slugs = get_ground_truth_slugs(query, df_train)
        
        if not truth_slugs:
            continue
        
        # Get predictions from API
        pred_slugs = get_predictions(query, use_reranking=use_reranking, k=max(k_values))
        
        query_recall = {}
        for k in k_values:
            recall = calculate_recall_at_k(pred_slugs, truth_slugs, k)
            recalls[k].append(recall)
            query_recall[f'recall@{k}'] = recall
        
        short_query = query[:50] + '...' if len(query) > 50 else query
        print(f"  [{i+1}/{len(queries)}] R@10: {query_recall['recall@10']:.2f} | {short_query}")
        
        query_results.append({
            'query': short_query,
            'ground_truth': len(truth_slugs),
            'use_reranking': use_reranking,
            **query_recall
        })
        
        # Small delay to avoid overwhelming API
        time.sleep(0.5)
    
    mean_recalls = {k: np.mean(recalls[k]) if recalls[k] else 0 for k in k_values}
    return mean_recalls, query_results


def main():
    print("=" * 60)
    print("LLM RERANKING EVALUATION")
    print("=" * 60)
    
    # Check if API is running
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ API not responding at {API_URL}")
            print("   Please start the API first: cd step5_api && uvicorn main:app --reload")
            return
    except:
        print(f"❌ Cannot connect to API at {API_URL}")
        print("   Please start the API first: cd step5_api && uvicorn main:app --reload")
        return
    
    print(f"✅ API connected at {API_URL}")
    
    # Load training data
    df_train = pd.read_excel('../train.xlsx')
    print(f"✅ Loaded {df_train['Query'].nunique()} unique queries\n")
    
    k_values = [1, 5, 10, 20]
    
    # Evaluate WITHOUT reranking
    recalls_baseline, results_baseline = evaluate(df_train, use_reranking=False, k_values=k_values)
    
    # Evaluate WITH reranking  
    recalls_llm, results_llm = evaluate(df_train, use_reranking=True, k_values=k_values)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs LLM Reranking")
    print("=" * 60)
    print(f"\n{'Metric':<12} | {'Baseline':<15} | {'LLM Rerank':<15} | {'Δ Change'}")
    print("-" * 60)
    
    for k in k_values:
        baseline = recalls_baseline[k]
        llm = recalls_llm[k]
        delta = llm - baseline
        delta_pct = (delta / baseline * 100) if baseline > 0 else 0
        sign = "+" if delta >= 0 else ""
        
        print(f"Recall@{k:<3}   | {baseline*100:>6.1f}%         | {llm*100:>6.1f}%         | {sign}{delta_pct:.1f}%")
    
    print("-" * 60)
    
    # Save results
    os.makedirs('../results/metrics', exist_ok=True)
    
    comparison_df = pd.DataFrame([
        {
            'K': k, 
            'Baseline': f"{recalls_baseline[k]*100:.1f}%",
            'LLM_Reranking': f"{recalls_llm[k]*100:.1f}%",
            'Delta': f"{(recalls_llm[k]-recalls_baseline[k])*100:+.1f}%"
        }
        for k in k_values
    ])
    comparison_df.to_csv('../results/metrics/llm_reranking_comparison.csv', index=False)
    
    print(f"\n✅ Saved to ../results/metrics/llm_reranking_comparison.csv")


if __name__ == "__main__":
    main()

