import os
import json
import time
import requests
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'raw_catalog.json')
BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
MAX_WORKERS = 10 
EXPECTED_PER_PAGE = 12 

TYPE_MAPPING = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

def get_details(item):
    """
    Worker function to scrape details from the Webpage HTML only.
    Finds the PDF URL but does NOT download it.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Initialize Defaults
    item['description'] = None
    item['duration'] = None
    item['test_type'] = None 
    item['job_level'] = None
    item['language'] = None  # <--- NEW FIELD
    item['pdf_url'] = None

    try:
        # Random sleep to prevent rate limiting
        time.sleep(random.uniform(0.1, 0.3))
        r = requests.get(item['url'], headers=headers, timeout=10)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html.parser')
            
            # 1. DESCRIPTION
            try:
                desc_header = soup.find('h4', string=re.compile("Description", re.IGNORECASE))
                if desc_header:
                    desc_p = desc_header.find_next_sibling('p')
                    if desc_p:
                        item['description'] = desc_p.get_text(strip=True)
            except: pass
            
            # 2. DURATION
            try:
                dur_header = soup.find('h4', string=re.compile("Assessment length", re.IGNORECASE))
                if dur_header:
                    dur_p = dur_header.find_next_sibling('p')
                    if dur_p:
                        text = dur_p.get_text(strip=True)
                        match = re.search(r'(\d+)', text)
                        if match:
                            item['duration'] = int(match.group(1))
            except: pass

            # 3. LANGUAGES (NEW)
            try:
                lang_header = soup.find('h4', string=re.compile("Languages", re.IGNORECASE))
                if lang_header:
                    lang_p = lang_header.find_next_sibling('p')
                    if lang_p:
                        raw_lang = lang_p.get_text(strip=True)
                        # Clean up trailing commas often found in this field
                        item['language'] = raw_lang.strip().strip(',')
            except: pass

            # 4. TEST TYPES
            try:
                found_specific_types = False
                paragraphs = soup.find_all('p', class_='product-catalogue__small-text')
                for p in paragraphs:
                    if "Test Type:" in p.get_text():
                        keys = p.find_all('span', class_='product-catalogue__key')
                        real_types = []
                        for k in keys:
                            code = k.get_text(strip=True)
                            if code in TYPE_MAPPING:
                                real_types.append(TYPE_MAPPING[code])
                        if real_types:
                            item['test_type'] = list(set(real_types))
                            found_specific_types = True
                        break 
                
                # Fallback to codes from the main table if detailed page extraction failed
                if not found_specific_types and 'test_type_codes' in item and item['test_type_codes']:
                    mapped_codes = []
                    for code in item['test_type_codes']:
                        if code in TYPE_MAPPING:
                            mapped_codes.append(TYPE_MAPPING[code])
                    if mapped_codes:
                        item['test_type'] = list(set(mapped_codes))
            except: pass

            # 5. JOB LEVEL
            try:
                job_header = soup.find('h4', string=re.compile("Job levels", re.IGNORECASE))
                if job_header:
                    job_p = job_header.find_next_sibling('p')
                    if job_p:
                        clean_level = job_p.get_text(strip=True).strip().strip(',')
                        if clean_level:
                            item['job_level'] = clean_level
            except: pass

            # 6. FIND PDF URL (NO DOWNLOAD)
            try:
                # Priority: Look for "Product Fact Sheet" link text
                pdf_link = soup.find('a', string=re.compile("Product Fact Sheet", re.IGNORECASE))
                
                # Fallback: Look for any link ending in .pdf
                if not pdf_link:
                    pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.IGNORECASE))

                if pdf_link:
                    pdf_url = pdf_link['href']
                    
                    # Fix Relative URLs
                    if not pdf_url.startswith('http'):
                        base = "https://service.shl.com"
                        if not pdf_url.startswith('/'): base += "/"
                        pdf_url = base + pdf_url
                    
                    # Fix Malformed URLs
                    if "https://service.shl.comhttps" in pdf_url:
                        pdf_url = pdf_url.replace("https://service.shl.comhttps", "https")

                    item['pdf_url'] = pdf_url
            except: pass

    except Exception:
        pass

    # Cleanup temporary field
    if 'test_type_codes' in item: del item['test_type_codes']
    
    return item

def scrape_shl():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("🚀 WEB SCRAPER: Starting Metadata Extraction (Languages Included)...")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') 
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-sh-usage')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_window_size(1920, 1080)
    driver.get(BASE_URL)
    
    all_links = []
    page_number = 1
    
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, ".NET")))
        time.sleep(2)

        while True:
            items_on_this_page = []
            attempts = 0
            
            # Retry loop to handle lazy loading
            while attempts < 4: 
                items_on_this_page = []
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 500);")
                time.sleep(1)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                rows = soup.find_all('tr')
                
                for row in rows:
                    try:
                        link_tag = row.find('a', href=True)
                        if not link_tag: continue
                        name = link_tag.get_text(strip=True)
                        url = link_tag['href']
                        if not url.startswith('http'): url = "https://www.shl.com" + url

                        if "Pre-packaged" in name: continue
                        
                        if any(x['url'] == url for x in all_links): continue
                        if any(x['url'] == url for x in items_on_this_page): continue

                        cols = row.find_all('td')
                        codes = []
                        if len(cols) > 0:
                            code_text = cols[-1].get_text(strip=True)
                            codes = list(code_text.replace(" ", ""))

                        items_on_this_page.append({
                            "name": name,
                            "url": url,
                            "test_type_codes": codes,
                        })
                    except: continue
                
                if len(items_on_this_page) >= EXPECTED_PER_PAGE:
                    break 
                elif page_number > 30: 
                    break
                else:
                    if attempts == 3:
                        print(f"⚠️ Page {page_number}: Accepting {len(items_on_this_page)} items.")
                        break
                    print(f"⚠️ Page {page_number}: Retrying ({attempts+1}/4)...")
                    time.sleep(2)
                    attempts += 1
            
            print(f"📄 Page {page_number}: collected {len(items_on_this_page)} items.")
            all_links.extend(items_on_this_page)

            # Click Next
            try:
                next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')]")
                if len(next_buttons) > 0:
                    target = next_buttons[-1]
                    if "disabled" in target.get_attribute("class"): 
                        print("✅ Reached last page.")
                        break
                    driver.execute_script("arguments[0].click();", target)
                    time.sleep(4) 
                    page_number += 1
                else: break
            except: break

    finally:
        driver.quit()

    print(f"\n🚀 Fetching Web Metadata for {len(all_links)} items...")

    final_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(get_details, item): item for item in all_links}
        completed = 0
        for future in as_completed(future_to_item):
            final_data.append(future.result())
            completed += 1
            if completed % 50 == 0:
                print(f"   [{completed}/{len(all_links)}] processed...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"🎉 DONE! Saved catalog to: {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_shl()