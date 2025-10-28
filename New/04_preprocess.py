#!/usr/bin/env python3
# 04_preprocess.py - FIXED DUPLICATE COLUMNS ISSUE
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- Load Data ---
df = pd.read_csv("/home/zubair/Desk/A/features_final.tsv", sep="\t", low_memory=False)

# --- Target Variable ---
if 'Is_Driver' in df.columns:
    df['is_driver'] = df['Is_Driver'].astype(int)
elif 'is_driver' not in df.columns:
    raise KeyError("No driver label column found. Expected 'Is_Driver' or 'is_driver'")

print(f"Class balance: {df['is_driver'].mean():.1%} drivers")

# --- Feature Selection ---
# Categorical features
cat_feats = [
    'Variant_Classification',
    'Cancer_Type',
    'callers',
    'Mutation_Type'
]
cat_feats = [c for c in cat_feats if c in df.columns]

# Numerical features (core)
num_core = [
    'AF',
    'cadd.phred' if 'cadd.phred' in df.columns else 'CADD_PHRED'
]
num_core = [c for c in num_core if c in df.columns]

# Signature probabilities and exposures
# First get all SBS columns then remove duplicates
all_sbs_cols = [c for c in df.columns if c.startswith('SBS')]
prob_cols = []
exp_cols = []

# Separate probabilities and exposures
for col in all_sbs_cols:
    if col.startswith('exp_SBS'):
        exp_cols.append(col)
    elif col not in exp_cols:  # Avoid duplicates
        prob_cols.append(col)

# Ensure unique features
num_feats = list(dict.fromkeys(num_core + prob_cols + exp_cols))  # Preserves order while removing duplicates

# Verify no duplicates
if len(num_feats) != len(set(num_feats)):
    duplicates = [x for x in num_feats if num_feats.count(x) > 1]
    raise ValueError(f"Duplicate features detected: {duplicates}")

# --- Train-Test Split ---
X = df[cat_feats + num_feats].copy()
y = df['is_driver']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Preprocessing ---
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_feats),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_feats)
])

# Fit and transform
print("\nFitting preprocessor...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# --- Save Outputs ---
output_dir = Path("/home/zubair/Desk/A/out")
output_dir.mkdir(exist_ok=True, parents=True)

joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
pd.DataFrame(X_train_proc).to_csv(output_dir / "X_train_proc.tsv", sep="\t", index=False)
pd.DataFrame(X_test_proc).to_csv(output_dir / "X_test_proc.tsv", sep="\t", index=False)
y_train.to_csv(output_dir / "y_train.tsv", sep="\t", index=False)
y_test.to_csv(output_dir / "y_test.tsv", sep="\t", index=False)

# --- Diagnostics ---
print("\n✅ Preprocessing complete")
print(f"Original features: {len(cat_feats + num_feats)}")
print(f"Processed features: {X_train_proc.shape[1]}")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Class balance (train): {y_train.mean():.1%} drivers")