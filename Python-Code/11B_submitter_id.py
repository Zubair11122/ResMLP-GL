import pandas as pd

base_dir = "C:/Users/Zubair/Desktop/AB/Final-project-files"
mut_file = f"{base_dir}/mutations_with_signatures_and_DP.tsv"
cancers = ["GBM", "COAD"]

mut_df = pd.read_csv(mut_file, sep="\t", dtype=str)

# Create a Sample ID column to match the clinical file format: first 15 chars
mut_df['Sample ID'] = mut_df['Tumor_Sample_Barcode'].str[:15].str.strip()

annotated = []

for cancer in cancers:
    print(f"\n--- Processing {cancer} ---")
    sample_meta_file = f"{base_dir}/{cancer.lower()}_tcga_gdc_clinical_data.tsv"
    samp_meta = pd.read_csv(sample_meta_file, sep="\t", dtype=str)
    samp_meta['Sample ID'] = samp_meta['Sample ID'].str.strip()
    samp_meta['Patient ID'] = samp_meta['Patient ID'].str.strip()

    df = mut_df[mut_df["Cancer_Type"] == cancer].copy()
    # Merge using the same (short) Sample ID format
    df = df.merge(
        samp_meta[["Sample ID", "Patient ID"]],
        on="Sample ID", how="inner"
    )
    df = df.rename(columns={"Patient ID": "case_id"})
    print(f"{cancer}: ▶️ {df['case_id'].nunique()} patients after sample‐merge")

    # Patient-level clinical
    clin = pd.read_csv(f"{base_dir}/clinical_{cancer}.tsv", sep="\t", dtype=str)
    clin.rename(columns={
        "cases.submitter_id": "case_id",
        "demographic.vital_status": "vital_status",
        "demographic.days_to_death": "days_to_death"
    }, inplace=True)

    # Follow-up
    fu = pd.read_csv(f"{base_dir}/follow_up_{cancer}.tsv", sep="\t", dtype=str)
    rename_map = {"cases.submitter_id": "case_id"}
    for col in [
        "follow_ups.days_to_last_follow_up",
        "follow_up.days_to_last_follow_up",
        "follow_ups.days_to_follow_up",
        "follow_up.days_to_follow_up"
    ]:
        if col in fu.columns:
            rename_map[col] = "days_to_last_follow_up"
            break
    fu.rename(columns=rename_map, inplace=True)
    fu["days_to_last_follow_up"] = pd.to_numeric(fu["days_to_last_follow_up"], errors="coerce")
    fu = fu[["case_id", "days_to_last_follow_up"]].groupby("case_id", as_index=False).max()

    # Merge clinical + follow-up
    clin = clin.merge(fu, on="case_id", how="left")
    clin["days_to_death"] = pd.to_numeric(clin["days_to_death"], errors="coerce")
    clin["time_days"] = clin["days_to_death"].fillna(clin["days_to_last_follow_up"])
    clin["Overall Survival (Months)"] = clin["time_days"] / 30.44
    clin["Overall Survival Status"] = (
        clin["vital_status"].str.lower().map({
            "dead": 1, "deceased": 1, "alive": 0, "living": 0
        }).fillna(0).astype(int)
    )

    # Merge survival back
    df = df.merge(
        clin[["case_id", "Overall Survival (Months)", "Overall Survival Status"]],
        on="case_id", how="left"
    )
    n_surv = df["Overall Survival Status"].notna().sum()
    print(f"{cancer}: ▶️ {n_surv} samples with survival data")
    annotated.append(df)

# Concatenate and write out
combined = pd.concat(annotated, ignore_index=True)
out_file = f"{base_dir}/mutations_with_clinical_combined.tsv"
combined.to_csv(out_file, sep="\t", index=False)
print(f"\n✅ Written combined file to {out_file}")
