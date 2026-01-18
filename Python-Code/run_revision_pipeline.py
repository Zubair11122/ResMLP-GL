import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter

########################################
# 1. LOAD YOUR FILES (LOW-MEMORY VERSION)
########################################

MUT_PATH = "C:/Users/Zubair/Desktop/AB/mutations_variant_complete.tsv"
GBM_PATH = "C:/Users/Zubair/Desktop/AB/gbm_tcga_gdc_clinical_data.tsv"
COAD_PATH = "C:/Users/Zubair/Desktop/AB/coad_tcga_gdc_clinical_data.tsv"

# First, read ONLY the header of mutation file to know which columns exist
mut_header = pd.read_csv(MUT_PATH, sep="\t", nrows=0)
all_cols = list(mut_header.columns)

# Columns we want (if they exist)
base_cols = [
    "sample", "hugo_symbol", "chromosome", "start_position", "end",
    "reference_allele", "tumor_seq_allele2", "tumor_sample_barcode",
    "amino_acid_change", "variant_classification", "callers",
    "cancer_type", "is_driver", "mutation", "context",
    "mutation_type", "type", "dominant_signature",
    "af", "dp", "so", "exonno", "cchange"
]

base_cols = [c for c in base_cols if c in all_cols]

# All SBS features
sbs_cols = [c for c in all_cols if c.lower().startswith("sbs")]

use_cols = base_cols + sbs_cols

print("Total columns in mutation file:", len(all_cols))
print("Columns loaded:", len(use_cols))

# Load mutation file in chunks
chunks = []
for chunk in pd.read_csv(
        MUT_PATH, sep="\t", usecols=use_cols,
        chunksize=30000, low_memory=True):
    chunks.append(chunk)

mut = pd.concat(chunks, ignore_index=True)
print("Mutations loaded:", mut.shape)

# Load clinical files
gbm = pd.read_csv(GBM_PATH, sep="\t")
coad = pd.read_csv(COAD_PATH, sep="\t")

########################################
# 2. PREPARE CLINICAL TABLES
########################################

gbm_clin = gbm[[
    "Patient ID", "Cancer Type",
    "Overall Survival (Months)",
    "Overall Survival Status"
]].rename(columns={
    "Patient ID": "case_id",
})

gbm_clin["Cancer_Type"] = "GBM"

coad_clin = coad[[
    "Patient ID", "Cancer Type",
    "Overall Survival (Months)",
    "Overall Survival Status"
]].rename(columns={
    "Patient ID": "case_id",
})

coad_clin["Cancer_Type"] = "COAD"

clinical = pd.concat([gbm_clin, coad_clin], ignore_index=True)
clinical = clinical.drop_duplicates(subset=["case_id"])

print("Clinical table:", clinical.shape)

########################################
# 3. EXTRACT PATIENT BARCODE FROM MUTATION SAMPLE
########################################

# sample like: TCGA-4W-AA9S-01A -> case_id: TCGA-4W-AA9S
mut["case_id"] = mut["sample"].str.slice(0, 12)

########################################
# 4. FILTER MUTATIONS TO ONLY CLINICAL PATIENTS
########################################

clinical_ids = set(clinical["case_id"].unique())
print("Unique clinical patient IDs:", len(clinical_ids))

# Keep only mutation rows whose case_id exists in clinical data
mut = mut[mut["case_id"].isin(clinical_ids)]

print("Mutations after filtering:", mut.shape)

########################################
# 5. MERGE MUTATIONS WITH CLINICAL SURVIVAL
########################################

mut2 = mut.merge(clinical, on="case_id", how="left")

########################################
# 6. IDENTIFY FEATURE COLUMNS
########################################

sbs_cols = [c for c in mut2.columns if c.lower().startswith("sbs")]
label_col = "is_driver"

non_feats = set([
    "sample", "hugo_symbol", "chromosome", "start_position", "end",
    "reference_allele", "tumor_seq_allele2", "tumor_sample_barcode",
    "amino_acid_change", "variant_classification", "callers",
    "mutation", "context", "mutation_type", "type",
    "dominant_signature", "so", "exonno", "cchange",
    "case_id", "Cancer_Type",
    "Overall Survival (Months)", "Overall Survival Status",
    "cancer_type"
])

feature_cols = [c for c in mut2.columns if c not in non_feats and c != label_col]
feature_cols_no_sbs = [c for c in feature_cols if c not in sbs_cols]

print("Feature columns:", len(feature_cols))
print("Feature columns no SBS:", len(feature_cols_no_sbs))

########################################
# 7. KEEP ONLY NUMERIC FEATURES
########################################

X_full = mut2[feature_cols].select_dtypes(include=[np.number]).fillna(0)
X_no_sbs = mut2[feature_cols_no_sbs].select_dtypes(include=[np.number]).fillna(0)

num_full = X_full.columns.tolist()
num_no_sbs = X_no_sbs.columns.tolist()

print("Numeric features (full):", len(num_full))
print("Numeric features (no SBS):", len(num_no_sbs))

########################################
# 8. LABEL VECTOR
########################################

y = mut2[label_col].replace(
    {True: 1, False: 0, "True": 1, "False": 0}
).fillna(0).astype(int)

########################################
# 9. TRAIN/TEST SPLIT + SCALING
########################################

Xp_f_train, Xp_f_test, yp_train, yp_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y)

Xp_n_train, Xp_n_test, yn_train, yn_test = train_test_split(
    X_no_sbs, y, test_size=0.2, random_state=42, stratify=y)

sc_f = StandardScaler()
Xp_f_train = sc_f.fit_transform(Xp_f_train)
Xp_f_test = sc_f.transform(Xp_f_test)

sc_n = StandardScaler()
Xp_n_train = sc_n.fit_transform(Xp_n_train)
Xp_n_test = sc_n.transform(Xp_n_test)

########################################
# 10. TRAIN MLP MODELS
########################################

mlp_full = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=40)
mlp_full.fit(Xp_f_train, yp_train)
pred_full = mlp_full.predict_proba(Xp_f_test)[:, 1]

mlp_nosbs = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=40)
mlp_nosbs.fit(Xp_n_train, yn_train)
pred_nosbs = mlp_nosbs.predict_proba(Xp_n_test)[:, 1]

auc_full = roc_auc_score(yp_test, pred_full)
auc_nosbs = roc_auc_score(yp_test, pred_nosbs)

print("\n===== ABLATION RESULTS =====")
print("Full model AUC:", auc_full)
print("No SBS AUC:", auc_nosbs)
print("ΔAUC:", auc_full - auc_nosbs)

########################################
# 11. BOOTSTRAP ΔAUC SIGNIFICANCE
########################################

def bootstrap_auc(y, p1, p2, n=500):
    diff = []
    N = len(y)
    for _ in range(n):
        idx = np.random.randint(0, N, N)
        d = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
        diff.append(d)
    diff = np.array(diff)
    return (
        diff.mean(),
        np.percentile(diff, 2.5),
        np.percentile(diff, 97.5),
        2 * min((diff > 0).mean(), (diff < 0).mean())
    )

mean_d, ci_l, ci_h, p_boot = bootstrap_auc(yp_test.values, pred_full, pred_nosbs)

print("\n===== BOOTSTRAP =====")
print("Mean ΔAUC:", mean_d)
print("95% CI:", ci_l, ci_h)
print("p-value:", p_boot)

########################################
# 12. DRIVER BURDEN PER PATIENT
########################################

X_full_all = mut2[num_full].fillna(0)
mut2["resmlp_prob"] = mlp_full.predict_proba(sc_f.transform(X_full_all))[:, 1]

burden = mut2.groupby("case_id")["resmlp_prob"].sum().reset_index()
burden = burden.rename(columns={"resmlp_prob": "driver_burden"})

surv = clinical.merge(burden, on="case_id", how="left")

surv = surv.rename(columns={
    "Overall Survival (Months)": "OS",
    "Overall Survival Status": "event"
})

# Make OS numeric
surv["OS"] = pd.to_numeric(surv["OS"], errors="coerce")

########################################
# 13. ROBUST PARSING OF SURVIVAL STATUS
########################################

def map_event(val):
    """Map survival status to 0/1:
       1 = death event, 0 = censored (alive)."""
    if pd.isna(val):
        return np.nan

    # Direct numeric
    if isinstance(val, (int, float)):
        if val in [0, 1]:
            return int(val)

    s = str(val).strip()

    # Patterns like '1:DECEASED', '0:LIVING'
    if s.startswith("1:") or "DECEASED" in s.upper() or "DEAD" in s.upper():
        return 1
    if s.startswith("0:") or "LIVING" in s.upper() or "ALIVE" in s.upper():
        return 0

    # Fallback for plain '1' / '0'
    if s == "1":
        return 1
    if s == "0":
        return 0

    return np.nan

surv["event"] = surv["event"].apply(map_event)

print("\nNon-null OS count:", surv["OS"].notna().sum())
print("Non-null event count:", surv["event"].notna().sum())
print("Non-null driver_burden count:", surv["driver_burden"].notna().sum())

########################################
# 14. COX SURVIVAL MODELS
########################################

cph = CoxPHFitter()

for cancer in ["GBM", "COAD"]:
    sdf = surv[surv["Cancer_Type"] == cancer].copy()
    sdf = sdf.dropna(subset=["OS", "event", "driver_burden"], how="any")

    print(f"\n===== COX MODEL: {cancer} =====")
    print("Rows:", len(sdf))

    if len(sdf) < 20:
        print("Not enough data for survival model.")
        continue

    cph.fit(
        sdf[["OS", "event", "driver_burden"]],
        duration_col="OS", event_col="event")
    print(cph.summary)

########################################
# 15. MUTATION FREQUENCY + CI
########################################

freq_g = (
    mut.groupby(["cancer_type", "hugo_symbol"])["sample"]
        .nunique()
        .reset_index()
        .rename(columns={"sample": "mutated_samples"})
)

total_samples = mut.groupby("cancer_type")["sample"] \
    .nunique().reset_index().rename(columns={"sample": "total_samples"})

freq = freq_g.merge(total_samples, on="cancer_type")
freq["freq"] = freq["mutated_samples"] / freq["total_samples"]

z = 1.96
freq["CI_low"] = freq["freq"] - z * np.sqrt(
    freq["freq"] * (1 - freq["freq"]) / freq["total_samples"])
freq["CI_high"] = freq["freq"] + z * np.sqrt(
    freq["freq"] * (1 - freq["freq"]) / freq["total_samples"])

freq.to_csv("C:/Users/Zubair/Desktop/AB/mutation_frequency_with_CI.csv", index=False)

print("\n===== ALL ANALYSES COMPLETE =====\n")
