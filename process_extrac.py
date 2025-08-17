import pandas as pd
import os

INPUT_CSV  = r"rule_functional_performance_qa.csv"      # has a 'question' column
OUTPUT_CSV = r"rule_functional_performance_qa_2.csv"

# Make sure the output directory exists (handles bare filenames too)
out_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV))
os.makedirs(out_dir, exist_ok=True)

# Read
df = pd.read_csv(INPUT_CSV)

# Extract rule number (e.g., V.3.2.10, AA.1.1.1, EV.7.5)
pattern = r'(?i)\brule\s+([A-Z]{1,4}(?:\.\d+)+)\b'   # anchored to the word "rule"
# If you prefer looser matching (any token like LETTERS.digits(.digits)*), use:
# pattern = r'\b([A-Z]{1,4}(?:\.\d+)+)\b'

# Update via .loc to avoid SettingWithCopyWarning
df.loc[:, 'rule_number'] = df['question'].str.extract(pattern)

# Save to a NEW file
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved: {OUTPUT_CSV}")

# (Optional) quick sanity check
matched = df['rule_number'].notna().sum()
print(f"Extracted rule numbers for {matched}/{len(df)} rows.")

# import pandas as pd
# import os

# INPUT_CSV  = r"rule_compilation_qa.csv"          # has a 'question' column
# OUTPUT_CSV = r"rule_compilation_qa_2.csv"

# # Ensure output folder exists (works for bare filenames too)
# os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)

# df = pd.read_csv(INPUT_CSV)

# # Grab ONLY the first backticked phrase as a STRING
# pattern = r'`([^`]+)`'
# df.loc[:, 'keyword_phrases_all'] = df['question'].str.extract(pattern).astype('string')

# df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
# print(f"Saved: {OUTPUT_CSV}")
