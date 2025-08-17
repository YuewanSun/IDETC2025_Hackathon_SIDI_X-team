import pandas as pd
import re, json, os
from typing import List, Dict, Any

CSV_PATH = r"C:\Users\ys25268\OneDrive - The University of Texas at Austin\Windows\Reaserch2\IDECT_hackathon\design_qa\dataset\rule_extraction\rule_compilation_qa.csv"
OUT_JSON = "extracted_keywords.json"

def detect_question_column(cols: List[str]) -> str:
    # Prefer exact 'question', else first column containing 'question'
    for c in cols:
        if c.lower() == "question":
            return c
    for c in cols:
        if "question" in c.lower():
            return c
    # common fallbacks
    for cand in ["prompt", "input", "query", "text"]:
        for c in cols:
            if c.lower() == cand:
                return c
    # if nothing obvious, default to the first column
    return cols[0]

def detect_id_column(cols: List[str]) -> str | None:
    for name in ["id", "row_id", "index", "uid", "qid"]:
        for c in cols:
            if c.lower() == name:
                return c
    return None

splitter = re.compile(r"\s*(?:/|,|;|\||\bor\b|\band\b)\s*", re.I)

def normalize_keyword(tok: str) -> str:
    t = tok.strip()
    # collapse internal whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def extract_keywords_from_question(q: str) -> Dict[str, Any]:
    if not isinstance(q, str):
        return {"raw": [], "keywords": []}
    # find all segments between backticks
    raw_segments = re.findall(r"`([^`]+)`", q)
    keywords: List[str] = []
    for seg in raw_segments:
        parts = [normalize_keyword(p) for p in splitter.split(seg) if p and p.strip()]
        keywords.extend(parts)
    # de-duplicate preserving order
    seen = set()
    deduped = []
    for k in keywords:
        kl = k  # keep original case
        if (kl.lower()) not in seen:
            seen.add(kl.lower())
            deduped.append(kl)
    return {"raw": raw_segments, "keywords": deduped}

# Load CSV
df = pd.read_csv(CSV_PATH, dtype=str)  # read as strings to be safe
q_col = detect_question_column(df.columns.tolist())
id_col = detect_id_column(df.columns.tolist())

per_row = []
all_keywords = []

for idx, row in df.iterrows():
    qtext = row.get(q_col, "")
    ex = extract_keywords_from_question(qtext)
    row_id = row.get(id_col, None) if id_col else None
    record = {
        "row_index": int(idx),
        **({"row_id": row_id} if row_id is not None else {}),
        "question": qtext,
        "raw": ex["raw"],
        "keywords": ex["keywords"],
    }
    per_row.append(record)
    all_keywords.extend(ex["keywords"])

# unique keywords across file
unique_keywords = []
seen_global = set()
for k in all_keywords:
    kl = k.lower()
    if kl not in seen_global:
        seen_global.add(kl)
        unique_keywords.append(k)

result = {"per_row": per_row, "unique_keywords": unique_keywords}

# Save JSON
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

