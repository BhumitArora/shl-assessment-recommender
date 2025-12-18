"""
Step 4: Evaluation Script

Evaluates the hybrid skill search algorithm on training data.
Calculates Recall@K for K = 1, 5, 10, 20

Usage:
    python evaluate_recall.py
    
Output:
    - Prints recall metrics
    - Saves results to ../results/metrics/recall_metrics.csv
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step3_retrieval.hybrid_skill_search import HybridSkillSearch


def normalize_url(url: str) -> str:
    """Extract slug from URL for matching."""
    return str(url).strip().lower().rstrip("/").split("/")[-1]


def get_ground_truth_indices(query: str, df_truth: pd.DataFrame, df_db: pd.DataFrame) -> list:
    """Get database indices for ground truth URLs."""
    urls = df_truth[df_truth['Query'] == query]['Assessment_url'].tolist()
    indices = []
    
    for url in urls:
        slug = normalize_url(url)
        match = df_db[df_db['url'].str.lower().str.contains(slug, na=False)]
        if not match.empty:
            indices.append(match.index[0])
    
    return indices


def evaluate(search_engine: HybridSkillSearch, df_train: pd.DataFrame, 
             df_db: pd.DataFrame, k_values: list = [1, 5, 10, 20]):
    """
    Evaluate recall at different K values.
    
    Returns:
        dict: {k: mean_recall} for each k
    """
    recalls = {k: [] for k in k_values}
    query_results = []
    
    for query in df_train['Query'].unique():
        truth_indices = get_ground_truth_indices(query, df_train, df_db)
        
        if not truth_indices:
            continue
        
        # Get predictions
        results = search_engine.search(query, k=max(k_values))
        pred_indices = [df_db[df_db['url'] == r['url']].index[0] 
                        for r in results if not df_db[df_db['url'] == r['url']].empty]
        
        query_recall = {}
        for k in k_values:
            preds_at_k = set(pred_indices[:k])
            hits = len(preds_at_k & set(truth_indices))
            recall = hits / len(truth_indices)
            recalls[k].append(recall)
            query_recall[f'recall@{k}'] = recall
        
        # Detect skills for reporting
        skills = [s[0] for s in search_engine.detect_skills(query)]
        
        query_results.append({
            'query': query[:60] + '...' if len(query) > 60 else query,
            'skills': ', '.join(skills[:4]),
            'ground_truth': len(truth_indices),
            **query_recall
        })
    
    return {k: np.mean(recalls[k]) for k in k_values}, query_results


def main():
    # Load data
    print("Loading data...")
    df_db = pd.read_csv('../data/processed_assessments.csv')
    df_db = df_db[df_db["rich_text"].notna()].reset_index(drop=True)
    df_train = pd.read_excel('../train.xlsx')
    
    print(f"Loaded {len(df_db)} assessments")
    print(f"Loaded {df_train['Query'].nunique()} unique queries from training set\n")
    
    # Initialize search engine
    search_engine = HybridSkillSearch(df_db)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    k_values = [1, 5, 10, 20]
    mean_recalls, query_results = evaluate(search_engine, df_train, df_db, k_values)
    
    # Print summary
    print("\nMean Recall@K:")
    print("-"*40)
    for k in k_values:
        bar = "█" * int(mean_recalls[k] * 30)
        print(f"  Recall@{k:<2}: {mean_recalls[k]:.4f} ({mean_recalls[k]*100:.1f}%) | {bar}")
    
    # Print per-query breakdown
    print("\n" + "="*60)
    print("PER-QUERY BREAKDOWN")
    print("="*60 + "\n")
    
    print(f"{'Query':<45} | {'Skills':<20} | {'GT':<3} | {'R@10'}")
    print("-"*85)
    
    for r in query_results:
        print(f"{r['query']:<45} | {r['skills']:<20} | {r['ground_truth']:<3} | {r['recall@10']:.2f}")
    
    # Save results
    os.makedirs('../results/metrics', exist_ok=True)
    
    # Save summary metrics
    metrics_df = pd.DataFrame([
        {'K': k, 'Mean_Recall': mean_recalls[k], 'Percentage': f"{mean_recalls[k]*100:.1f}%"}
        for k in k_values
    ])
    metrics_df.to_csv('../results/metrics/recall_metrics.csv', index=False)
    
    # Save per-query results
    query_df = pd.DataFrame(query_results)
    query_df.to_csv('../results/metrics/per_query_recall.csv', index=False)
    
    print(f"\n✓ Saved metrics to ../results/metrics/")
    print(f"  - recall_metrics.csv")
    print(f"  - per_query_recall.csv")
    
    return mean_recalls


if __name__ == "__main__":
    main()

