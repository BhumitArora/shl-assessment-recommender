import os
import json

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_catalog.json')

def enrich_data():
    print(f"Loading data from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Error: raw_catalog.json not found. Run scraper first.")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} items...")
    
    for item in data:
        # 1. Clean up fields to ensure they are strings
        name = item.get('name', 'Unknown Assessment')
        
        # Join list of types into a string "Type1, Type2"
        types = ", ".join(item.get('test_type', []))
        if not types:
            types = "General"
            
        level = item.get('job_level', 'General Level')
        desc = item.get('description', '')

        # 2. Create the "Golden Combination" String
        # Format: "Name [Types] for Level: Description"
        # This puts the most important keywords (Name/Type) at the start.
        combined_text = f"{name} [{types}] for {level}: {desc}"
        
        # 3. Save it into the object
        item['search_context'] = combined_text

    # 4. Save back to file
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("🎉 Success! Added 'search_context' to all items.")
    print("Example Entry:")
    print(f"Name: {data[0]['name']}")
    print(f"Context: {data[0]['search_context'][:150]}...")

if __name__ == "__main__":
    enrich_data()