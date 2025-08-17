import pandas as pd
import os
import ast
from typing import List, Dict, Optional
import torch
from typing import Optional, List
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# Global variables for model caching
model = None
processor = None

def load_model_if_needed(model_id: str):
    """Load model and processor only once, cache globally."""
    global model, processor
    
    if model is None or processor is None:
        print("Loading model for the first time...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully.")
    else:
        print("Using cached model.")

def add_context_to_qa(
    chunks_csv: str,
    qa_csv: str,
    output_csv: str,
    *,
    chunk_id_col: Optional[str] = None,   # e.g., "chunk_id"
    text_col: Optional[str] = None,       # e.g., "chunk_text"
    qa_list_col: str = "chunk_id_list",   # in the QA csv
    sep: str = "\n\n"                     # how to join chunk texts
) -> pd.DataFrame:
    """
    Join chunk texts from `chunks_csv` based on ID lists in `qa_csv`, write a new
    'context' column to `output_csv`, and return the updated QA DataFrame.

    The `qa_list_col` may contain values like "20,21,32" or "[20, 21, 32]".
    If `chunk_id_col`/`text_col` are not provided, the function tries common names.
    """

    # --- helpers ---
    def pick_col(df: pd.DataFrame, provided: Optional[str], candidates: List[str], kind: str) -> str:
        if provided and provided in df.columns:
            return provided
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Couldn't find a {kind} column. Tried: { [provided] + candidates } in {list(df.columns)}")

    def parse_id_list(s) -> List[int]:
        """Parse '20,21,32' or '[20, 21, 32]' → [20, 21, 32]."""
        if pd.isna(s):
            return []
        s = str(s).strip()
        if not s:
            return []
        # Try literal list first
        if s.startswith('[') and s.endswith(']'):
            try:
                vals = ast.literal_eval(s)
                return [int(v) for v in vals]
            except Exception:
                pass
        # Fallback: comma-separated
        s = s.strip("[]")
        out = []
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                # ignore non-integers
                pass
        return out

    # --- load ---
    df_chunks = pd.read_csv(chunks_csv)
    df_qa = pd.read_csv(qa_csv)

    # --- pick cols (auto-detect if not provided) ---
    chunk_id_col = pick_col(df_chunks, chunk_id_col, ["chunk_id"], "chunk_id")
    text_col     = pick_col(df_chunks, text_col,     ["text"], "text")
    if qa_list_col not in df_qa.columns:
        raise ValueError(f"'{qa_list_col}' not found in QA CSV columns: {list(df_qa.columns)}")

    # --- build id→text map (cast ids to int; keep rows with valid ids only) ---
    ids_int = pd.to_numeric(df_chunks[chunk_id_col], errors="coerce")
    valid = ids_int.notna()
    chunk_map: Dict[int, str] = dict(
        zip(ids_int[valid].astype(int), df_chunks.loc[valid, text_col].astype(str))
    )

    # --- create context column (preserve order in the list; skip missing ids) ---
    df_qa.loc[:, "context"] = df_qa[qa_list_col].apply(
        lambda s: sep.join(chunk_map[i] for i in parse_id_list(s)[:11] if i in chunk_map)
    )
    # --- save ---
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_qa.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved with context → {output_csv}")

    return df_qa

def build_chat_text(q: str, ctx: str, system_prompt: str) -> str:
    """
    Build a chat-formatted prompt using the processor's chat template.
    """
    # Ensure strings
    q = "" if pd.isna(q) else str(q)
    ctx = "" if pd.isna(ctx) else str(ctx)

    user_text = (
        "Use the following CONTEXT to answer the QUESTION.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{q}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    # Get a *string* prompt; we'll tokenize together for batching
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def generate_single(text: str, max_new_tokens: int, temperature: float, do_sample: bool) -> str:
    """
    Generate response for a single chat-formatted string.
    Returns decoded, prompt-free model output.
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    # Remove the prompt (get only new tokens)
    input_length = inputs["input_ids"].shape[1]
    gen_tokens = gen[0, input_length:]
    text = processor.decode(gen_tokens, skip_special_tokens=True).strip()
    return text

def run_llama_on_table(
    input_csv: str,
    output_csv: str,
    model_id: str,
    *,
    question_col: Optional[str] = None,   # auto-detects if None
    context_col: str = "context",
    system_prompt: str = "You are a helpful assistant. Use the provided context to answer the question concisely.",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> pd.DataFrame:
    """
    Reads a CSV with question/context columns, gets model responses row by row,
    writes to output_csv, and returns the updated DataFrame.

    Args:
        input_csv: Path to input CSV. Must contain 'question' (or 'qustion') and 'context'.
        output_csv: Path to save CSV with new 'model_prediction' column.
        model_id: HF model path or repo id (e.g., local Llama 3.2 Vision Instruct).
        question_col: Column name for questions; if None, auto-detects 'question' or 'qustion'.
        context_col: Column name for context.
        system_prompt: System instruction used in the chat template.
        max_new_tokens: Max tokens to generate per example.
        temperature: Sampling temperature (0.0 for deterministic).
        do_sample: Whether to sample.

    Returns:
        pd.DataFrame with an added 'model_prediction' column.
    """
    # ---- Load model once ----
    load_model_if_needed(model_id)
    
    # ---- I/O ----
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df = pd.read_csv(input_csv)

    # --- pick question column (handle common misspelling) ---
    if question_col is None:
        if "question" in df.columns:
            question_col = "question"
        else:
            raise ValueError(f"Could not find a question column ('question' or 'qustion') in {list(df.columns)}")

    if context_col not in df.columns:
        raise ValueError(f"Context column '{context_col}' not found. Available: {list(df.columns)}")

    # ---- Process row by row ----
    preds: List[str] = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            # Build chat text for this row
            chat_text = build_chat_text(
                row[question_col], 
                row[context_col], 
                system_prompt
            )
            
            # Generate response
            prediction = generate_single(
                chat_text,
                max_new_tokens,
                temperature,
                do_sample
            )
            print(f"Row {idx}: Generated prediction: {prediction[:50]}...")  # Print first 50 chars
            preds.append(prediction)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            preds.append(f"[ERROR] {e}")

    # ---- save & return ----
    df["model_prediction"] = preds
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved predictions → {output_csv}")
    return df

# Example usage:
if __name__ == "__main__":
    # add_context_to_qa(
    #     chunks_csv="rules_chunks_processed_2.csv",
    #     qa_csv="rule_compilation_qa_2_with_predictions_3.csv",
    #     output_csv="rule_compilation_qa_2_with_predictions_3_with_context.csv",
    #     # chunk_id_col="chunk_id",       # (optional) set explicitly if needed
    #     # text_col="chunk_text",         # (optional) set explicitly if needed
    #     qa_list_col="chunk_id_list",
    #     sep="\n\n"
    # )
    system_prompt = (
        "You are an expert of rules for the FSAE competition. Only give rule number list, no other words, like:['T.7', 'T.7.1', 'T.7.1.1', 'T.7.1.3']"
    )
    run_llama_on_table(
        input_csv=r"rule_compilation_qa_2_with_predictions_3_with_context.csv",
        output_csv=r"rule_compilation_qa_answer_2.csv",
        model_id=r"C:\Software\Model\llama3\Llama-3.2-11B-Vision-Instruct",
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        system_prompt=system_prompt
    )