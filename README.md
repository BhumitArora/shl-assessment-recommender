# Assessment Recommendation System

An intelligent recommendation engine that suggests relevant assessments based on job descriptions or natural language queries.

## Live Demo

| Resource | URL |
|----------|-----|
| **API** | https://shl-assessment-recommender-i3m5.onrender.com |
| **Frontend** | https://shl-assessment-recommender-six.vercel.app |
| **GitHub** | https://github.com/BhumitArora/shl-assessment-recommender |

## Performance

| Approach | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|----------|----------|----------|-----------|-----------|
| TF-IDF Baseline | 0.0% | 18.0% | 28.0% | 35.0% |
| + Query Expansion | 2.0% | 25.0% | 38.0% | 42.0% |
| + Skill Detection | 2.0% | 32.7% | 47.7% | 47.7% |
| + LLM Reranking | 4.0% | 35.2% | **50.5%** | 51.0% |

## Project Structure

```
├── step1_data_ingestion/       # Scraping & data collection
├── step2_preprocessing/        # Cleaning & embeddings
├── step3_retrieval/            # Search algorithms
│   ├── tfidf_baseline.py       # TF-IDF keyword search
│   └── hybrid_skill_search.py  # Multi-skill hybrid search
├── step4_evaluation/           # Metrics & predictions
│   ├── evaluate_recall.py      # Recall@K calculation
│   └── generate_predictions.py # Test set predictions
├── step5_api/                  # FastAPI application
│   └── main.py                 # API server
├── frontend/                   # Web interface
│   └── index.html              # Vercel-hosted UI
├── data/                       # Assessment database
│   ├── processed_assessments.csv
│   └── assessment_embeddings_google.npy
└── results/                    # Output files
    ├── predictions/
    └── metrics/
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
cd step5_api
uvicorn main:app --port 8000

# Evaluate model
cd step4_evaluation
python evaluate_recall.py
```

## API Endpoints

### Health Check
```
GET /health
```
```json
{"status": "healthy"}
```

### Get Recommendations
```
POST /recommend
Content-Type: application/json

{"query": "Python developer with SQL experience"}
```

Response:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/products/product-catalog/view/python-new/",
      "name": "Python (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test that measures knowledge of Python programming...",
      "duration": 11,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

## Approach

### 1. Skill Detection
Regex-based pattern matching to identify skills in queries:
- Technical: Python, SQL, Java, JavaScript, Excel
- Cognitive: Reasoning, Aptitude, Verify
- Personality: OPQ, Behavioral assessments
- Business: Sales, Marketing, Leadership

### 2. Query Expansion
Role-based keyword mappings to enrich queries:
```
"Java" → "Java JVM J2EE Spring Hibernate backend"
"Sales" → "Sales revenue client acquisition CRM"
```

### 3. Hybrid Search
Combined TF-IDF keyword matching with skill-based slot allocation:
- High priority skills get more recommendation slots
- Ensures coverage of all detected skills

### 4. LLM Reranking (Optional)
LangChain + Gemini 2.0 Flash for semantic relevance scoring.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python |
| Search | TF-IDF, Scikit-learn |
| LLM | LangChain, Gemini |
| Frontend | HTML/CSS/JS |
| Deployment | Render, Vercel |

## Files

| File | Description |
|------|-------------|
| `train.xlsx` | 10 training queries with ground truth |
| `test-set.xlsx` | 9 test queries |
| `test_predictions.csv` | Final predictions for submission |
| `requirements.txt` | Python dependencies |

## License

MIT
