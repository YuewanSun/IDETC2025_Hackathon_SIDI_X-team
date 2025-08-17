import pandas as pd
import os

INPUT_CSV  = r"rule_functional_performance_qa_2.csv"                 # <-- change me
OUTPUT_CSV = r"rule_functional_performance_qa_2.csv"    # <-- change me

# Ensure output folder exists (works for bare filenames too)
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Find the prediction column (handles both 'model_prediction' and 'model prediction')
col_candidates = ["model_prediction", "model prediction"]
pred_col = next((c for c in col_candidates if c in df.columns), None)
if pred_col is None:
    raise ValueError(f"Couldn't find prediction column in {list(df.columns)}. "
                     f"Tried {col_candidates}.")

def add_answer_prefix(x):
    if pd.isna(x):
        return x
    s = str(x)
    # avoid double-prefixing
    if s.lstrip().lower().startswith("answer:"):
        return s
    return f"answer: {s}"

# Use .loc to avoid SettingWithCopyWarning
df.loc[:, pred_col] = df[pred_col].apply(add_answer_prefix)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved: {OUTPUT_CSV}")
