import pandas as pd

# 1) Load your full TSV
df = pd.read_csv("mutations_with_signatures_and_DP.tsv", sep="\t")

# 2) Use the exact column names from your file
cancer_col = "Cancer_Type"     # e.g. "GBM", "LUAD"
driver_col = "Is_Driver"       # should be 1/0 or True/False per row

# If Is_Driver is stored as strings "True"/"False", uncomment:
# df[driver_col] = df[driver_col].astype(str).str.lower().map({'true': True, 'false': False})

# 3) Sampling parameters
cancer_types         = ["GBM", "LUAD"]
drivers_per_type     = 60
non_drivers_per_type = 60

# 4) Sample rows for each cancer type
chunks = []
for ct in cancer_types:
    sub = df[df[cancer_col] == ct]
    
    # driver vs non‑driver rows
    drv_rows = sub[sub[driver_col] == 1]       # or == True
    non_rows = sub[sub[driver_col] == 0]       # or == False
    
    # sample (will error if fewer than 60—adjust replace=True if needed)
    s_drv = drv_rows.sample(n=drivers_per_type, random_state=42)
    s_non = non_rows.sample(n=non_drivers_per_type, random_state=42)
    
    chunks.append(pd.concat([s_drv, s_non]))

# 5) Build the final balanced DataFrame (should be 240 rows)
balanced = pd.concat(chunks).sample(frac=1, random_state=42).reset_index(drop=True)
print("Final shape:", balanced.shape)  # → (240, number_of_columns)

# 6) Export to TSV
balanced.to_csv("balanced_240_variants.tsv", sep="\t", index=False)
