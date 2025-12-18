import os
import json
import time
import requests
import fitz  # PyMuPDF
from google import genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'raw_catalog.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'enriched_catalog_llm.json')

# --- PRODUCTION SETTINGS ---
# 20 seconds = 3 requests/minute. 
# This ensures we stay UNDER the limit rather than hitting it and waiting.
DELAY_BETWEEN_REQUESTS = 20  

# Setup Client
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Error: GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)

# --- MODEL SELECTION ---
# Using the specific 2.5 Flash model requested
MODEL_NAME = 'gemini-2.5-flash' 

def get_pdf_text(pdf_url):
    """Downloads PDF and extracts text."""
    if not pdf_url: return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
            
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception:
        return None

def extract_with_retry(pdf_text):
    """Calls Gemini LLM with infinite retry logic."""
    if not pdf_text or len(pdf_text) < 50: return None

    prompt = f"""
    You are a data extraction assistant. Extract details from this SHL Assessment Fact Sheet into JSON.
    
    Extract these exact keys (return null if missing):
    1. "Assessment Name": Title of the assessment.
    2. "Overview": Overview summary text.
    3. "Relevant Job Roles": Job roles text.
    4. "Language": Language (e.g. English US).
    5. "Average Testing Time": Value for 'Average Testing Time'.
    6. "Allowed Time": Value for 'Allowed Time'.
    7. "Max Questions": Value for 'Maximum Number of Questions'.
    8. "Sittings": Value for 'Number of Sittings'.
    9. "Test Type": Value for 'Test Type'.
    10. "Sector": Value for 'Sector'.
    11. "Scores Reported": Value for 'Scores Reported'.
    12. "ONET Competency": Value for 'O*NET Competency'.
    13. "Competencies Measured": Array of strings from 'Knowledge, Skills...' section.

    Return ONLY valid JSON.
    
    --- START TEXT ---
    {pdf_text[:15000]}
    --- END TEXT ---
    """
    
    # Start with a higher wait time to clear the "penalty box" immediately
    wait_time = 30 
    
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            
            raw_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_json)
            
        except Exception as e:
            error_str = str(e).lower()
            # Catch 429 (Rate Limit) & 503 (Overloaded)
            if "429" in error_str or "resource exhausted" in error_str or "quota" in error_str or "503" in error_str:
                print(f"\n      ⚠️ Quota Limit Hit. Pausing for {wait_time}s to reset...", end="", flush=True)
                time.sleep(wait_time)
                print(" Retrying...")
                # Cap the wait time at 2 minutes so we don't wait forever
                wait_time = min(wait_time * 2, 120)
            
            # Catch Model Not Found (404) specifically for new models
            elif "404" in error_str and "model" in error_str:
                 print(f"\n      ❌ Error: Model '{MODEL_NAME}' not found or not available in your region yet.")
                 print("      💡 Try switching back to 'gemini-1.5-flash' if this persists.")
                 return None
                 
            else:
                print(f"      ❌ Non-recoverable Error: {e}")
                return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Resume logic
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 Resuming from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        print(f"📂 Loading fresh data from {INPUT_FILE}...")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Filter items needing processing
    items_to_process = [
        item for item in data 
        if item.get('pdf_url') and not item.get('pdf_data')
    ]
    
    total = len(data)
    remaining = len(items_to_process)
    
    print(f"🚀 Processing Remaining: {remaining} items")
    print(f"🐢 Safe Mode: 1 request every {DELAY_BETWEEN_REQUESTS} seconds")
    print(f"🤖 Model: {MODEL_NAME}")
    print("="*50)

    for i, item in enumerate(items_to_process):
        print(f"[{i+1}/{remaining}] Processing: {item['name'][:40]}...", end=" ", flush=True)

        raw_text = get_pdf_text(item['pdf_url'])
        
        if raw_text:
            llm_data = extract_with_retry(raw_text)
            if llm_data:
                idx = data.index(item)
                data[idx]['pdf_data'] = llm_data
                print("✅")
            else:
                print("❌ LLM Error")
        else:
            print("❌ PDF Error")

        # Save immediately
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        # Don't sleep after the very last item
        if i < remaining - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\n" + "="*50)
    print("🎉 FULL CATALOG PROCESSED!")
    
if __name__ == "__main__":
    main()