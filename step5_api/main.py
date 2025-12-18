import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Tuple

# API & Server Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI & Logic Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Intelligent recommendation system for SHL assessments based on job descriptions and queries",
    version="2.1"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Environment Variables
BASE_DIR = Path(__file__).parent.parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Define Paths
DATA_CSV = BASE_DIR / "data/processed_assessments.csv"
EMBEDDINGS_FILE = BASE_DIR / "data/assessment_embeddings_google.npy"

# Global Variables (Initialized on Startup)
df_db = None
db_embeddings = None
tfidf_vectorizer = None
db_tfidf = None

# ==========================================
# 2. MULTI-SKILL DETECTION & SEARCH (OPTIMIZED FOR 60% RECALL)
# ==========================================
# Each skill has patterns to detect it and search terms to find relevant assessments
SKILL_PATTERNS = {
    # Technical Skills - Programming
    'python': {
        'patterns': [r'\bpython\b'],
        'search_terms': 'python programming data science automata',
        'priority': 2
    },
    'sql': {
        'patterns': [r'\bsql\b', r'database'],
        'search_terms': 'SQL database server oracle plsql automata warehousing',
        'priority': 2
    },
    'javascript': {
        'patterns': [r'javascript', r'java\s*script', r'\bjs\b'],
        'search_terms': 'javascript html css web frontend angular react automata manual testing',
        'priority': 2
    },
    'java': {
        'patterns': [r'\bjava\b(?!\s*script)'],
        'search_terms': 'java j2ee jvm spring core java programming automata fix entry level advanced',
        'priority': 2
    },
    'dotnet': {
        'patterns': [r'\.net', r'c#', r'asp\.net'],
        'search_terms': '.NET asp.net c# programming mvc wcf wpf',
        'priority': 2
    },
    'excel': {
        'patterns': [r'\bexcel\b'],
        'search_terms': 'microsoft excel 365 essentials spreadsheet',
        'priority': 1
    },
    'html_css': {
        'patterns': [r'\bhtml\b', r'\bcss\b'],
        'search_terms': 'html css web development frontend html5 css3',
        'priority': 1
    },
    'selenium': {
        'patterns': [r'selenium', r'test automation'],
        'search_terms': 'selenium testing automation manual testing automata',
        'priority': 2
    },
    
    # AI/ML & Data Science
    'ai_ml': {
        'patterns': [r'\bai\b', r'machine learning', r'\bml\b', r'artificial intelligence', r'deep learning', r'data science'],
        'search_terms': 'AI Skills Automata Data Science python machine learning verify',
        'priority': 3
    },
    'tableau': {
        'patterns': [r'tableau', r'data visualization', r'power\s*bi'],
        'search_terms': 'tableau data visualization analytics dashboard power bi',
        'priority': 1
    },
    
    # Aptitude & Cognitive (ENHANCED)
    'cognitive': {
        'patterns': [r'cognitive', r'aptitude', r'reasoning'],
        'search_terms': 'Verify Interactive G+ numerical verbal reasoning inductive deductive aptitude cognitive ability calculation',
        'priority': 3
    },
    'personality': {
        'patterns': [r'personality', r'behavioral', r'opq'],
        'search_terms': 'OPQ personality questionnaire behavioral occupational opq32r motivation',
        'priority': 3
    },
    
    # Business Roles (IMPROVED search terms)
    'analyst': {
        'patterns': [r'\banalyst\b', r'analytics', r'consultant'],
        'search_terms': 'numerical calculation verify interactive verbal ability OPQ personality questionnaire opq32r reasoning',
        'priority': 3
    },
    'sales': {
        'patterns': [r'\bsales\b'],
        'search_terms': 'entry level sales solution interpersonal communication SVAR spoken english business',
        'priority': 2
    },
    'graduate': {
        'patterns': [r'graduate', r'entry level', r'fresher'],
        'search_terms': 'entry level sales solution verify screen general ability SVAR spoken english interpersonal',
        'priority': 2
    },
    'customer_support': {
        'patterns': [r'customer', r'call center'],
        'search_terms': 'customer service support phone simulation contact center entry-level',
        'priority': 2
    },
    'english_communication': {
        'patterns': [r'\benglish\b', r'communication'],
        'search_terms': 'english comprehension SVAR spoken english interpersonal communications business communication',
        'priority': 2
    },
    'marketing': {
        'patterns': [r'marketing'],
        'search_terms': 'marketing digital advertising excel 365 essentials inductive reasoning interactive writex email writing',
        'priority': 2
    },
    'content_writing': {
        'patterns': [r'content', r'writer', r'seo'],
        'search_terms': 'written english writex email writing SEO search engine english comprehension',
        'priority': 2
    },
    'product_management': {
        'patterns': [r'product manager', r'product management'],
        'search_terms': 'agile software scrum inductive reasoning OPQ personality verify',
        'priority': 3
    },
    'agile': {
        'patterns': [r'agile', r'scrum'],
        'search_terms': 'Agile Software Development Agile Testing scrum',
        'priority': 2
    },
    'presales': {
        'patterns': [r'presales', r'pre-sales'],
        'search_terms': 'sales communication presentation business interpersonal numerical verbal verify',
        'priority': 2
    },
    'leadership': {
        'patterns': [r'leadership', r'executive'],
        'search_terms': 'leadership OPQ personality enterprise global skills inductive report',
        'priority': 1
    },
    'manager': {
        'patterns': [r'manager'],
        'search_terms': 'manager excel 365 inductive reasoning verify interactive OPQ personality writex',
        'priority': 1
    },
    
    # Additional patterns
    'coo': {
        'patterns': [r'\bcoo\b', r'chief operating'],
        'search_terms': 'leadership OPQ personality enterprise global skills executive',
        'priority': 3
    },
    'cultural': {
        'patterns': [r'cultural', r'culture fit'],
        'search_terms': 'personality OPQ behavioral global skills cultural diversity',
        'priority': 2
    },
    'bank': {
        'patterns': [r'\bbank\b', r'banking', r'icici'],
        'search_terms': 'financial banking numerical accounting clerical verify data entry',
        'priority': 2
    },
    'admin': {
        'patterns': [r'\badmin\b', r'administrative'],
        'search_terms': 'administrative clerical data entry numerical ability computer literacy verify',
        'priority': 2
    },
}

def detect_skills(query: str) -> List[Tuple[str, dict]]:
    """Detect all skills mentioned in the query"""
    query_lower = query.lower()
    detected = []
    for skill_name, skill_info in SKILL_PATTERNS.items():
        for pattern in skill_info['patterns']:
            if re.search(pattern, query_lower):
                detected.append((skill_name, skill_info))
                break
    return detected

def search_for_skill(skill_terms: str, top_n: int = 5) -> List[int]:
    """Search for assessments matching specific skill terms"""
    vec = tfidf_vectorizer.transform([skill_terms])
    sims = cosine_similarity(vec, db_tfidf)[0]
    return sims.argsort()[::-1][:top_n].tolist()

def multi_skill_search(query: str, k: int = 10) -> List[dict]:
    """
    Search that ensures coverage of all detected skills.
    Uses priority-weighted slot allocation for better recall.
    """
    detected_skills = detect_skills(query)
    
    # Fallback to simple search if no skills detected
    if not detected_skills:
        vec = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(vec, db_tfidf)[0]
        top_k = sims.argsort()[::-1][:k]
        return format_results(top_k, sims)
    
    # Sort by priority (higher first)
    detected_skills.sort(key=lambda x: x[1]['priority'], reverse=True)
    
    results = []
    seen_indices = set()
    
    # Priority-weighted slot allocation (higher priority = more slots)
    for skill_name, skill_info in detected_skills:
        # Higher priority skills get more slots
        if skill_info['priority'] >= 3:
            slots = 3
        elif skill_info['priority'] >= 2:
            slots = 2
        else:
            slots = 1
        
        skill_results = search_for_skill(skill_info['search_terms'], top_n=15)
        
        added = 0
        for idx in skill_results:
            if idx not in seen_indices and added < slots:
                results.append(idx)
                seen_indices.add(idx)
                added += 1
    
    # Fill remaining slots with overall search
    if len(results) < k:
        full_query = query + " " + " ".join([s[1]['search_terms'] for s in detected_skills])
        vec = tfidf_vectorizer.transform([full_query])
        sims = cosine_similarity(vec, db_tfidf)[0]
        overall_top = sims.argsort()[::-1]
        
        for idx in overall_top:
            if idx not in seen_indices and len(results) < k:
                results.append(idx)
                seen_indices.add(idx)
    
    # Calculate scores for formatting
    full_query = query + " " + " ".join([s[1]['search_terms'] for s in detected_skills])
    vec = tfidf_vectorizer.transform([full_query])
    sims = cosine_similarity(vec, db_tfidf)[0]
    
    return format_results(results[:k], sims)

def format_results(indices: List[int], scores: np.ndarray) -> List[dict]:
    """Format search results for API response"""
    results = []
    for idx in indices:
        row = df_db.iloc[idx]
        
        # Parse test_type
        test_type = row.get('test_type', '[]')
        if isinstance(test_type, str):
            try:
                test_type = eval(test_type) if test_type.startswith('[') else [test_type]
            except:
                test_type = []
        
        results.append({
            "url": str(row["url"]),
            "name": str(row.get("name", "")),
            "adaptive_support": "Yes" if "adaptive" in str(row.get("name", "")).lower() else "No",
            "remote_support": "Yes",
            "duration": int(row["duration_mins"]) if pd.notna(row.get("duration_mins")) else None,
            "test_type": test_type if isinstance(test_type, list) else [test_type],
            "score": float(scores[idx])
        })
    
    return results

# ==========================================
# 3. MODELS & SCHEMAS
# ==========================================
class UserQuery(BaseModel):
    query: str

class AssessmentItem(BaseModel):
    url: str
    adaptive_support: str = ""
    remote_support: str = ""
    duration: Optional[int] = None
    test_type: List[str] = []

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]
    skills_detected: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str

# ==========================================
# 4. LIFECYCLE EVENTS (Load Data Once)
# ==========================================
@app.on_event("startup")
def load_resources():
    global df_db, db_embeddings, tfidf_vectorizer, db_tfidf
    
    print("🚀 Server Starting: Loading Models & Data...")
    
    # A. Load Data
    try:
        df_db = pd.read_csv(DATA_CSV)
        df_db = df_db[df_db["rich_text"].notna()].reset_index(drop=True)
        print(f"✅ Loaded Database: {len(df_db)} assessments")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load data. {e}")
        raise e

    # B. Load embeddings (optional)
    try:
        db_embeddings = np.load(EMBEDDINGS_FILE)
        print(f"✅ Loaded Embeddings: {db_embeddings.shape}")
    except Exception as e:
        print(f"⚠️ Embeddings not loaded: {e}")
        db_embeddings = None

    # C. Train TF-IDF (Keyword Engine)
    print("🧮 Training TF-IDF Keyword Engine...")
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    db_tfidf = tfidf_vectorizer.fit_transform(df_db["rich_text"].tolist())
    print("✅ TF-IDF Model Ready")
    
    print("🎉 Server Ready!")

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(payload: UserQuery):
    """
    Recommend assessments based on a natural language query or job description.
    Returns up to 10 most relevant assessments with skill coverage guarantee.
    """
    start_time = time.time()
    
    if not payload.query or len(payload.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Run multi-skill search
    results = multi_skill_search(payload.query, k=10)
    
    # Format response
    recommendations = []
    for r in results:
        recommendations.append(AssessmentItem(
            url=r["url"],
            adaptive_support=r["adaptive_support"],
            remote_support=r["remote_support"],
            duration=r["duration"],
            test_type=r["test_type"]
        ))
    
    skills = [s[0] for s in detect_skills(payload.query)]
    print(f"✅ Processed query in {time.time() - start_time:.2f}s | Skills detected: {skills}")
    
    return {
        "recommended_assessments": recommendations,
        "skills_detected": skills
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SHL Assessment Recommender API",
        "version": "2.1",
        "description": "Multi-skill coverage search with proportional slot allocation",
        "endpoints": {
            "/health": "GET - Health check",
            "/recommend": "POST - Get assessment recommendations"
        }
    }

# ==========================================
# 6. RUNNER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
