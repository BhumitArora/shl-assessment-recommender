"""
Step 4: Generate Predictions for Test Set

Generates assessment recommendations for test queries and saves to CSV.

Usage:
    python generate_predictions.py
    
Output:
    - ../results/predictions/test_predictions.csv
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step3_retrieval.hybrid_skill_search import HybridSkillSearch


def main():
    # Load data
    print("="*60)
    print("GENERATING TEST SET PREDICTIONS")
    print("="*60 + "\n")
    
    df_db = pd.read_csv('../data/processed_assessments.csv')
    df_db = df_db[df_db["rich_text"].notna()].reset_index(drop=True)
    df_test = pd.read_excel('../test-set.xlsx')
    
    print(f"Loaded {len(df_db)} assessments")
    print(f"Loaded {len(df_test)} test queries\n")
    
    # Initialize search engine
    search_engine = HybridSkillSearch(df_db)
    
    # Generate predictions
    predictions = []
    
    for i, query in enumerate(df_test['Query'].tolist()):
        skills = [s[0] for s in search_engine.detect_skills(query)]
        
        print(f"\nQuery {i+1}: {query[:50]}...")
        print(f"  Skills: {skills}")
        
        results = search_engine.search(query, k=10)
        
        print(f"  Top 3 recommendations:")
        for j, r in enumerate(results[:3], 1):
            print(f"    {j}. {r['name']}")
        
        for r in results:
            predictions.append({
                'Query': query,
                'Assessment_url': r['url']
            })
    
    # Save predictions
    os.makedirs('../results/predictions', exist_ok=True)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv('../results/predictions/test_predictions.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {len(df_predictions)} predictions ({len(df_test)} queries × 10 recommendations)")
    print(f"✓ Saved to ../results/predictions/test_predictions.csv")
    print("="*60)
    
    # Also copy to root for submission
    df_predictions.to_csv('../test_predictions.csv', index=False)
    print(f"✓ Also saved to ../test_predictions.csv (for submission)")


if __name__ == "__main__":
    main()

