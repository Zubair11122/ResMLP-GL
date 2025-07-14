import os, sys
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Repro folder & timestamp ──────────────────────────────────────────────────
OUT_DIR = "C:/Users/Zubair/Desktop/AB/Final-project-files/shap_results"
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 1. Load data and labels ───────────────────────────────────────────────────
print("⏳ Loading mutation table …")
df = pd.read_csv("C:/Users/Zubair/Desktop/AB/Final-project-files/mutations_variant_complete.tsv", sep="\t")
df.replace("-", np.nan, inplace=True)

if "is_driver" not in df.columns:
    sys.exit("ERROR: 'is_driver' column not found in mutations_variant_complete.tsv")

y = df["is_driver"].astype(int)

# ── 2. Apply the saved scikit-learn pre-processor ─────────────────────────────
print("🔧 Applying preprocessing pipeline …")
preproc = joblib.load("C:/Users/Zubair/Desktop/AB/Final-project-files/preprocessor.pkl")

X_raw = df[preproc.feature_names_in_]      # keep only expected columns
X = preproc.transform(X_raw)               # numpy array
feat_out = preproc.get_feature_names_out() # names after encoding / scaling

# keep a DataFrame wrapper (handy for SHAP sampling)
X_df = pd.DataFrame(X, columns=feat_out)

# ── 3. Load trained Keras model ───────────────────────────────────────────────
print("🧠 Loading trained model …")
model = tf.keras.models.load_model("C:/Users/Zubair/Desktop/AB/Final-project-files/driver_prediction_model.keras")

# ── 4. SHAP – speed-friendly settings ─────────────────────────────────────────
print("🔍 Computing SHAP values …")
SAMPLE_ROWS = min(500, len(X_df))
STRAT_SAMPLE, _ = train_test_split(
    X_df,
    test_size=len(X_df) - SAMPLE_ROWS,
    stratify=y,
    random_state=42,
)

BACKGROUND = shap.sample(STRAT_SAMPLE, 50)
expl = shap.KernelExplainer(model.predict, BACKGROUND, link="logit")

try:
    shap_vals = expl.shap_values(STRAT_SAMPLE, nsamples=150)
except Exception as e:
    sys.exit(f"SHAP failed: {e}")

# harmonise list / ndarray output
if isinstance(shap_vals, list):            # binary classifier
    shap_vals = np.sum(np.abs(np.array(shap_vals)), axis=0)
else:
    shap_vals = np.abs(np.array(shap_vals))

mean_abs_shap = shap_vals.mean(axis=0).flatten()   # mean |SHAP| per encoded feat

# ── 5. Map encoded names back to original column names ────────────────────────
print("🔗 Mapping SHAP scores to original variables …")
orig_cols = df.columns.tolist()            # names in the raw TSV

def orig_var(encoded: str) -> str:
    """
    Recover the raw column that a transformed / one-hot feature came from,
    by finding the **longest raw column name that is a prefix**.
    Fall back to the encoded name if nothing matches.
    """
    matches = [c for c in orig_cols if encoded.startswith(c)]
    return max(matches, key=len) if matches else encoded

orig_names = [orig_var(f) for f in feat_out]

# build aggregated SHAP table
shap_df = (
    pd.DataFrame({"orig": orig_names, "shap": mean_abs_shap})
      .groupby("orig", as_index=False)["shap"].sum()
)

# ✂️  Remove technical prefixes added by ColumnTransformer
shap_df["orig"] = shap_df["orig"].str.replace(r"^(num|cat|ord|bin)__", "", regex=True)

# ── 6.  Top-20 bar chart ──────────────────────────────────────────────────────
top20 = shap_df.sort_values("shap", ascending=False).head(20)

print("📈 Plotting top-20 variables …")
plt.figure(figsize=(10, 7))
plt.barh(range(20, 0, -1), top20["shap"], color="steelblue")
plt.yticks(range(20, 0, -1), top20["orig"])
plt.xlabel("Mean |SHAP| value")
plt.title("Top-20 important variables")
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, f"top20_{stamp}.png")
plt.savefig(plot_path, dpi=900)
plt.close()

# ── 7.  Save CSV of the 20 variables ─────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, f"top20_{stamp}.csv")
top20.to_csv(csv_path, index=False)

# ── 8.  Done ─────────────────────────────────────────────────────────────────
print(f"""
✅ SHAP analysis complete
   •  Chart : {plot_path}
   •  CSV   : {csv_path}

Top-5 variables:
{top20.head(5).to_string(index=False)}
""")
