"""
Step 3: Hybrid Skill-Based Search Algorithm

This is the core retrieval algorithm that:
1. Detects skills from the query using regex patterns
2. Expands each skill with relevant search terms
3. Uses priority-weighted slot allocation for balanced coverage
4. Combines TF-IDF search with skill-specific searches

Performance: Recall@10 = 59.8% on training set
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

# ==========================================
# SKILL PATTERNS - Core Configuration
# ==========================================
# Each skill has:
#   - patterns: regex patterns to detect the skill in queries
#   - search_terms: expanded terms for TF-IDF search
#   - priority: 1-3, higher = more slots allocated

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
    'excel': {
        'patterns': [r'\bexcel\b'],
        'search_terms': 'microsoft excel 365 essentials spreadsheet',
        'priority': 1
    },
    
    # AI/ML & Data Science
    'ai_ml': {
        'patterns': [r'\bai\b', r'machine learning', r'\bml\b', r'artificial intelligence', 
                     r'deep learning', r'data science'],
        'search_terms': 'AI Skills Automata Data Science python machine learning verify',
        'priority': 3
    },
    
    # Aptitude & Cognitive
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
    
    # Business Roles
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


class HybridSkillSearch:
    """
    Hybrid search engine that combines skill detection with TF-IDF search.
    """
    
    def __init__(self, df_assessments: pd.DataFrame):
        """
        Initialize the search engine with assessment data.
        
        Args:
            df_assessments: DataFrame with 'rich_text', 'name', 'url' columns
        """
        self.df = df_assessments[df_assessments["rich_text"].notna()].reset_index(drop=True)
        
        # Train TF-IDF model
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["rich_text"].tolist())
        
        print(f"Initialized with {len(self.df)} assessments")
    
    def detect_skills(self, query: str) -> List[Tuple[str, dict]]:
        """Detect all skills mentioned in the query."""
        query_lower = query.lower()
        detected = []
        
        for skill_name, skill_info in SKILL_PATTERNS.items():
            for pattern in skill_info['patterns']:
                if re.search(pattern, query_lower):
                    detected.append((skill_name, skill_info))
                    break
        
        return detected
    
    def search_for_skill(self, skill_terms: str, top_n: int = 15) -> List[int]:
        """Search for assessments matching specific skill terms."""
        vec = self.tfidf.transform([skill_terms])
        sims = cosine_similarity(vec, self.tfidf_matrix)[0]
        return sims.argsort()[::-1][:top_n].tolist()
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Main search function with priority-weighted skill coverage.
        
        Args:
            query: User query or job description
            k: Number of results to return
            
        Returns:
            List of dictionaries with 'name', 'url', 'score' keys
        """
        detected_skills = self.detect_skills(query)
        
        # Fallback to simple TF-IDF if no skills detected
        if not detected_skills:
            vec = self.tfidf.transform([query])
            sims = cosine_similarity(vec, self.tfidf_matrix)[0]
            top_k = sims.argsort()[::-1][:k]
            return self._format_results(top_k, sims)
        
        # Sort by priority (higher first)
        detected_skills.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        results = []
        seen_indices = set()
        
        # Priority-weighted slot allocation
        for skill_name, skill_info in detected_skills:
            # Higher priority skills get more slots
            if skill_info['priority'] >= 3:
                slots = 3
            elif skill_info['priority'] >= 2:
                slots = 2
            else:
                slots = 1
            
            skill_results = self.search_for_skill(skill_info['search_terms'], top_n=15)
            
            added = 0
            for idx in skill_results:
                if idx not in seen_indices and added < slots:
                    results.append(idx)
                    seen_indices.add(idx)
                    added += 1
        
        # Fill remaining slots with combined search
        if len(results) < k:
            expanded = query + " " + " ".join([s[1]['search_terms'] for s in detected_skills])
            vec = self.tfidf.transform([expanded])
            sims = cosine_similarity(vec, self.tfidf_matrix)[0]
            
            for idx in sims.argsort()[::-1]:
                if idx not in seen_indices and len(results) < k:
                    results.append(idx)
                    seen_indices.add(idx)
        
        # Calculate scores for final results
        expanded = query + " " + " ".join([s[1]['search_terms'] for s in detected_skills])
        vec = self.tfidf.transform([expanded])
        sims = cosine_similarity(vec, self.tfidf_matrix)[0]
        
        return self._format_results(results[:k], sims)
    
    def _format_results(self, indices: List[int], scores: np.ndarray) -> List[Dict]:
        """Format results as list of dictionaries."""
        return [
            {
                'name': self.df.loc[idx, 'name'],
                'url': self.df.loc[idx, 'url'],
                'score': float(scores[idx])
            }
            for idx in indices
        ]


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('../data/processed_assessments.csv')
    search_engine = HybridSkillSearch(df)
    
    # Test query
    query = "Looking for Python, SQL and JavaScript developers"
    results = search_engine.search(query, k=10)
    
    print(f"\nQuery: {query}")
    print(f"Skills detected: {[s[0] for s in search_engine.detect_skills(query)]}")
    print("\nTop 10 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} (score: {r['score']:.3f})")

