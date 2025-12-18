import json
import pandas as pd
import re


# ---------- helpers ----------
def clean_duration_to_int(val):
    """
    Extracts the first number from a string (e.g. "30 mins" -> 30)
    Returns 0 if no number found.
    """
    if pd.isna(val) or val == "":
        return 0
    # Find all numbers in the string
    numbers = re.findall(r'\d+', str(val))
    if numbers:
        return int(numbers[0])
    return 0

def safe_join(value):
    """
    Safely convert list / string / None into a clean string
    """
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if isinstance(value, str):
        return value
    return ""


def safe_str(value):
    """
    Convert value to string only if meaningful
    """
    if value is None:
        return ""
    return str(value)


# ---------- core logic ----------

def create_rich_text(row, debug=False):
    """
    Build embedding-ready rich text from one row
    """
    text = ""

    # Basic metadata
    text += f"Assessment Name: {safe_str(row.get('name'))}. "
    text += f"Description: {safe_str(row.get('description'))}. "
    text += f"Job Level: {safe_str(row.get('job_level'))}. "
    text += f"Language: {safe_str(row.get('language'))}. "

    # Test type
    text += f"Test Type: {safe_join(row.get('test_type'))}. "

    # Duration
    if row.get("duration") not in (None, 0, "0"):
        text += f"Duration: {safe_str(row.get('duration'))} minutes. "

    # ---------- PDF data ----------
    details = row.get("pdf_data")

    if isinstance(details, dict) and details:
        text += f"Overview: {safe_str(details.get('Overview'))}. "
        text += f"Allowed Time: {safe_str(details.get('Allowed Time'))}. "
        text += f"Max Questions: {safe_str(details.get('Max Questions'))}. "
        text += f"Sittings: {safe_str(details.get('Sittings'))}. "
        text += f"Sector: {safe_str(details.get('Sector'))}. "

        competencies = details.get("Competencies Measured")
        if competencies:
            text += f"Competencies: {safe_join(competencies)}. "

        scores = details.get("Scores Reported")
        if scores:
            text += f"Scores Reported: {safe_join(scores)}. "

        onet = details.get("ONET Competency")
        if onet:
            text += f"ONET Competency: {safe_str(onet)}. "

    else:
        if debug:
            print(f"[WARN] Missing or empty pdf_data for: {row.get('name')}")

    return text.strip()


# ---------- runner ----------

def main():
    INPUT_JSON = "/Users/bhumitarora/Desktop/Auto_assessment_recommender/data/enriched_catalog_llm.json"
    OUTPUT_CSV = "/Users/bhumitarora/Desktop/Auto_assessment_recommender/data/processed_assessments.csv"

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Create rich text column
    df["rich_text"] = df.apply(create_rich_text, axis=1)

    # Optional: drop empty rich text rows
    df = df[df["rich_text"].str.len() > 0]

    # Apply the rich text creation (Keep this, it's good)
    df["rich_text"] = df.apply(create_rich_text, axis=1)

    # --- ADD THIS SECTION ---
    # Create a pure integer column for filtering
    df["duration_mins"] = df["duration"].apply(clean_duration_to_int)
    # ------------------------

    # Save
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Processed {len(df)} rows")
    print(f"📁 Output saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
