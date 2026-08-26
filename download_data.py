import pandas as pd
from datasets import load_dataset

print("Downloading dataset from Hugging Face...")
# Loads the top 200 coding problem pairs
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:1000]")

df = pd.DataFrame(dataset)

# CodeSense requires 'instruction' and 'code' columns
clean_df = pd.DataFrame({
    "instruction": df["instruction"],
    "code": df["output"]
})

# Save directly to the CodeSense folder as a CSV
output_path = "custom_dataset.csv"
clean_df.to_csv(output_path, index=False)

print(f"Success! Saved {len(clean_df)} snippets to {output_path}")