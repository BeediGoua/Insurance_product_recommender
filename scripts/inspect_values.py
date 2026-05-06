import pandas as pd
from pathlib import Path

# Load data
df = pd.read_parquet('artifacts/baseline_v0/train_cleaned.parquet')

cols_to_check = ['occupation_code', 'branch_code', 'marital_status']

print("\n" + "="*50)
for col in cols_to_check:
    print(f"\n--- {col} ---")
    uniques = df[col].dropna().unique()
    print(f"Nombre de valeurs uniques : {len(uniques)}")
    print("Valeurs (Top 50) :")
    # Sort and print
    print(sorted(uniques.tolist())[:50]) 
    if len(uniques) > 50:
        print("... (liste tronquée)")
print("\n" + "="*50)
