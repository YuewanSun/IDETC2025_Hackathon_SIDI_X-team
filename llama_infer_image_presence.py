"""
Batch inference over a table of (prompt, image) using Llama 3.2 11B Vision Instruct.

Requirements (install once):
  pip install --upgrade transformers accelerate torch pandas pillow tqdm

Edit INPUT_CSV, OUTPUT_CSV, and MODEL_ID as needed.
The CSV must have columns: prompt, image  (image = file path to the image)
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

# -----------------------------
# Paths & config
# -----------------------------
MODEL_ID = r"C:\Software\Model\llama3\Llama-3.2-11B-Vision-Instruct"
INPUT_CSV = r"rule_presence_qa.csv"       # must have columns: prompt, image
OUTPUT_CSV = r"rule_presence_qa_2.csv"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
DO_SAMPLE = False

# Global variables for model caching
model = None
processor = None

# -----------------------------
# Load model & processor (only once)
# -----------------------------
def load_model_if_needed():
    global model, processor
    
    if model is None or processor is None:
        print("Loading model for the first time...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("Model loaded successfully.")
    else:
        print("Using cached model.")

# -----------------------------
# Helper functions
# -----------------------------
def build_chat_text(user_prompt: str) -> str:
    """
    Uses the model's chat template (preferred) to format a vision+text prompt.
    """
    chat = [
        {"role": "system", "content": "You are an expert of vehicles for the FSAE competition."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]
    # tokenize=False so we get a string back, which we will tokenize together with image
    return processor.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )

def run_inference(prompt: str, image_path: str) -> str:
    # Ensure model is loaded
    load_model_if_needed()
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Build chat-formatted text and tokenize with image
    text = build_chat_text(prompt)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Generate - pass all necessary inputs for vision model
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,  # This includes input_ids, attention_mask, pixel_values, etc.
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

    # Decode - get only the new tokens (skip the input)
    input_token_len = inputs['input_ids'].shape[1]
    response_tokens = generated_ids[0][input_token_len:]
    out = processor.decode(response_tokens, skip_special_tokens=True).strip()
    return out

# -----------------------------
# Read table and infer
# -----------------------------
# Load model once before processing
load_model_if_needed()

df = pd.read_csv(INPUT_CSV)

image_folder = r"C:\Users\ys25268\OneDrive - The University of Texas at Austin\Windows\Reaserch2\IDECT_hackathon\design_qa\dataset\rule_comprehension\rule_presence_qa"
outputs = []
user_constrain="Only reply Yes or No, do not include any other text"
for i, row in tqdm(df.iterrows(), total=len(df), desc="Inferring"):
    prompt = row["question"] + "\n" + user_constrain
    image_path = os.path.join(image_folder, str(row["image"]))
    #print(image_path)
    try:
        response = run_inference(prompt, image_path)
        print(f"Response for row {i}: {response}")
    except Exception as e:
        response = f"[ERROR] {e}"

    outputs.append(response)

df["model_prediction"] = outputs
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Done. Saved results to: {OUTPUT_CSV}")