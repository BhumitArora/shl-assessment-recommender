# SHL Assessment Recommender

An intelligent recommendation system that suggests relevant SHL assessments based on natural language queries or job descriptions.

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Recall@5** | 44.7% |
| **Recall@10** | 59.8% |
| **Recall@20** | 70.2% |

## 🗂️ Project Structure

```
Auto_assessment_recommender/
│
├── step1_data_ingestion/          # Data collection & scraping
│   ├── scrape_shl.py              # Scrape SHL product catalog
│   ├── enrich_data.py             # Enrich with additional metadata
│   └── pdf_processing_llm.py      # Extract data from PDFs using LLM
│
├── step2_preprocessing/           # Data preprocessing & indexing
│   ├── preprocess_json.py         # Clean and preprocess JSON data
│   └── embeddings_generate.py     # Generate embeddings using Google API
│
├── step3_retrieval/               # Search & retrieval algorithms
│   ├── hybrid_skill_search.py     # ⭐ Core hybrid search algorithm
│   ├── tfidf_baseline.py          # TF-IDF baseline search
│   └── embedding_search.py        # Embedding-based semantic search
│
├── step4_evaluation/              # Evaluation & prediction scripts
│   ├── evaluate_recall.py         # Calculate Recall@K metrics
│   └── generate_predictions.py    # Generate test set predictions
│
├── step5_api/                     # FastAPI application
│   └── main.py                    # API server with /recommend endpoint
│
├── data/                          # Data files
│   ├── processed_assessments.csv  # Main assessment database
│   ├── assessment_embeddings_google.npy  # Pre-computed embeddings
│   └── *.json                     # Raw and enriched catalog data
│
├── results/                       # Output results
│   ├── predictions/               # Generated predictions
│   │   └── test_predictions.csv   # Test set predictions
│   ├── metrics/                   # Evaluation metrics
│   │   ├── recall_metrics.csv     # Summary Recall@K
│   │   └── per_query_recall.csv   # Per-query breakdown
│   └── visualizations/            # Plots and charts
│
├── train.xlsx                     # Training data (10 queries)
├── test-set.xlsx                  # Test data (9 queries)
└── test_predictions.csv           # Final predictions for submission
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv shl
source shl/bin/activate
pip install pandas numpy scikit-learn fastapi uvicorn openpyxl
```

### 2. Evaluate the Model
```bash
cd step4_evaluation
python evaluate_recall.py
```

### 3. Generate Predictions
```bash
cd step4_evaluation
python generate_predictions.py
```

### 4. Start the API Server
```bash
cd step5_api
uvicorn main:app --reload --port 8000
```

## 🔧 Algorithm Overview

### Hybrid Skill-Based Search

The core algorithm (`step3_retrieval/hybrid_skill_search.py`) combines:

1. **Skill Detection**: Regex-based pattern matching to identify skills in queries
2. **Query Expansion**: Each skill is mapped to relevant search terms
3. **Priority-Weighted Allocation**: High-priority skills get more recommendation slots
4. **TF-IDF Search**: Expanded queries are matched against assessment descriptions

### Skill Patterns

| Category | Skills |
|----------|--------|
| **Technical** | Python, SQL, JavaScript, Java, Excel |
| **AI/ML** | Machine Learning, Data Science, AI |
| **Cognitive** | Reasoning, Aptitude, Verify G+ |
| **Personality** | OPQ, Behavioral, Motivation |
| **Business** | Sales, Marketing, Customer Support |
| **Management** | Leadership, Product Management, Agile |

## 📈 Results

### Recall@K Summary

| K | Mean Recall |
|---|-------------|
| 1 | 3.3% |
| 5 | 44.7% |
| **10** | **59.8%** |
| 20 | 70.2% |

### Per-Query Performance

| Query Type | Recall@10 | Notes |
|------------|-----------|-------|
| Java Developers | 80% | Strong pattern match |
| Graduate Sales | 40% | Entry-level coverage |
| COO/Cultural | 67% | Leadership + OPQ |
| Content Writer | 60% | SEO + English |
| Bank Admin | 100% | Perfect match |
| Data Analyst | 50% | Multi-skill coverage |

## 🌐 API Usage

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Looking for Python and SQL developers"}'
```

### Response Format
```json
{
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "https://www.shl.com/products/product-catalog/view/python-new/",
      "duration": 40,
      "test_type": "Knowledge & Skills"
    }
  ],
  "debug_info": {
    "detected_skills": ["python", "sql"],
    "processing_time": "0.15s"
  }
}
```

## 📝 Files

- **train.xlsx**: 10 training queries with ground truth assessments
- **test-set.xlsx**: 9 test queries for prediction
- **test_predictions.csv**: Final predictions (10 per query)

## 🔮 Future Improvements

1. Add semantic search with embeddings
2. Implement LLM-based query understanding
3. Add duration-based filtering
4. Re-ranking with cross-encoder models

---

*Built for SHL AI Intern Generative AI Assignment*

