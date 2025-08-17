import re
import json
import torch
import pandas as pd
import unicodedata
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor
from ftfy import fix_text

RULE_PAT = re.compile(r'\b[A-Z]{1,3}\.?\d+(?:\.\d+){0,4}\b')

def clean_rule_text(s: str, bullets="newline"):
    """
    bullets: 'keep' | 'newline' | 'dash' | 'remove'
    """
    # 1) Repair mojibake and weird quotes/dashes etc.
    s = fix_text(s)
    # 2) Normalize unicode
    s = unicodedata.normalize("NFKC", s)
    # 3) Normalize whitespace
    s = s.replace("\u00A0", " ")               # non-breaking space -> space
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)

    # 4) Handle bullets
    if bullets == "newline":
        s = s.replace("•", "\n- ")
    elif bullets == "dash":
        s = s.replace("•", " - ")
    elif bullets == "remove":
        s = s.replace("•", " ")

    # 5) Optional: ensure space after periods like "EV.5.1.2Each" -> "EV.5.1.2 Each"
    s = re.sub(r"([A-Z]{1,3}\.\d+(?:\.\d+){0,4})(?=[A-Za-z])", r"\1 ", s)

    return s


def safe_last_json_block(text: str) -> dict | None:
    """
    Extract the last well-formed top-level JSON object from text.
    """
    stack = 0
    start = -1
    last = None
    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0 and start != -1:
                candidate = text[start:i+1].strip()
                try:
                    last = json.loads(candidate)
                except Exception:
                    pass
    return last

def extract_keywords_and_rules_batch(chunks_texts: list, model=None, processor=None) -> list:
    """
    Process multiple chunks at once with a single model instance.
    
    Args:
        chunks_texts: List of text chunks to process
        model: Pre-loaded model (optional)
        processor: Pre-loaded processor (optional)
        
    Returns:
        List of dictionaries with rule_numbers and keywords for each chunk
    """
    # Load model and processor if not provided
    if model is None or processor is None:
        model_id = r"C:\Software\Model\llama3\Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
    
    results = []
    
    # Process each chunk
    for chunk_text in chunks_texts:
        # 2) JSON-only instruction (no extra prose)
        user_prompt = f"""
Return ONLY valid JSON with this schema:
{{
  "rule_numbers": ["GR.1.2.3", "GR.4.5.6", ...],
  "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}

Rules text:
{chunk_text}

Instructions:
- Extract every rule identifier found (patterns like VE.2.5.1, EV.5.2, GR.1.2.3, etc.).
- Extract 5–15 important technical keywords/phrases from the text (no duplicates).
- Output only the JSON object. No comments, no explanations.
"""

        messages = [
            {"role": "system", "content": "You output only JSON that matches the schema. No extra text."},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        # 3) Tokenize with chat template and generate
        chat_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,     # inserts the assistant prefix
            tokenize=False                  # <-- return a string, not a tensor
        )

        # Now tokenize to a dict (input_ids, attention_mask)
        enc = processor(text=chat_text, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=384,
                do_sample=False,
                temperature=0.0,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        gen_only = generated[0, enc["input_ids"].shape[-1]:]
        raw_text = processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        
        # 5) Parse JSON robustly
        obj = None
        try:
            # common case: model returns pure JSON
            obj = json.loads(raw_text)
        except Exception:
            # fallback: find the last JSON block in case of stray tokens
            obj = safe_last_json_block(raw_text)

        if isinstance(obj, dict):
            obj.setdefault("rule_numbers", [])
            obj.setdefault("keywords", [])
            # Final light cleanup: dedupe/normalize
            obj["rule_numbers"] = sorted(set(obj["rule_numbers"]))
            obj["keywords"] = sorted({k.strip() for k in obj["keywords"] if k.strip()})
            results.append(obj)
            continue

        # 6) Last-resort fallback (regex + naive keywording) if the model didn't give JSON
        rules = sorted(set(RULE_PAT.findall(chunk_text)))
        # super-simple keywording: top capitalized/mid-length tokens + domain words
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", chunk_text)
        candidates = []
        stop = {"the","and","for","with","that","must","from","into","made","used","than","less","more",
                "are","any","all","not","top","each","between","into","about","after","before","this",
                "also","have","has","was","were","will","may","can","shall"}
        for t in tokens:
            tt = t.strip("-").lower()
            if tt not in stop and 3 <= len(tt) <= 20:
                candidates.append(tt)
        # keep some domain hints if present
        domain_boost = {
        "Aerodynamic",
        "Aerodynamics",
        "Tractive System",
        "Shutdown System",
        "Accelerator Pedal Position Sensor",
        "APPS",
        "Brake Pedal",
        "Suspension",
        "Battery",
        "Chassis",
        "Primary Structure",
        "Critical Fasteners",
        "Critical Fastener",
        "Envelope",
        "Tube",
        "Tubing",
        "Tubes",
        "Material properties",
        "material",
        "materials",
        "External Items",
        "External Item",
        "Impact Attenuator",
        "Accumulator",
        "Firewall",
        "Powertrain",
        "Catch Cans",
        "Thermal Protection",
        "Scatter Shields",
        "Coolant",
        "Butt Joints",
        "Butt Joint",
        "Inertia Switch",
        "Transponder",
        "Brake Over Travel Switch",
        "BOTS",
        "Wiring",
        "Grounded Low Voltage",
        "GLV",
        "Grounding",
        "Lighting",
        "Light",
        "Lights"
    }
        kws = list(sorted(set([w for w in candidates if w in domain_boost])))

        results.append({"rule_numbers": rules, "keywords": kws[:12]})
    
    return results

def process_chunks_csv(input_file, output_file, batch_size=5):
    """
    Process all chunks in the CSV file, extract keywords and rule numbers,
    and save the results to a new CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        batch_size: Number of chunks to process in each batch
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initialize new columns
    df['rule_numbers'] = None
    df['keywords'] = None
    
    # Load model and processor ONCE
    print("Loading model (this may take a minute)...")
    model_id = r"C:\Software\Model\llama3\Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Process in batches
    total_rows = len(df)
    print(f"Processing {total_rows} chunks in batches of {batch_size}...")
    
    for start_idx in tqdm(range(0, total_rows, batch_size)):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Get text chunks for this batch
        batch_texts = [clean_rule_text(text) for text in batch_df['text']]
        
        # Process batch
        batch_results = extract_keywords_and_rules_batch(batch_texts, model, processor)
        
        # Update dataframe with results
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            if i < len(batch_results):  # Safety check
                df.at[idx, 'rule_numbers'] = json.dumps(batch_results[i]['rule_numbers'])
                df.at[idx, 'keywords'] = json.dumps(batch_results[i]['keywords'])
    
    print(f"Saving processed data to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Processing complete!")

# --- quick test ---
if __name__ == "__main__":
    input_file = "../processed_dataset/rules_chunks.csv"
    output_file = "../processed_dataset/rules_chunks_processed_2.csv"
    process_chunks_csv(input_file, output_file, batch_size=5)