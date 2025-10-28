#!/usr/bin/env python3
# 03_build_feature_table.py - FINAL WORKING VERSION
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUTS = {
    "mutations": Path("/home/zubair/Desk/A/mutations_with_signatures_and_DP.tsv"),
    "opencravat": Path("/home/zubair/Desk/A/variant.csv"),
    "spa_prob": Path(
        "/home/zubair/Desk/A/spa_out_restricted/Assignment_Solution/Activities/Decomposed_MutationType_Probabilities.txt"),
    "spa_stats": Path(
        "/home/zubair/Desk/A/spa_out_restricted/Assignment_Solution/Solution_Stats/Assignment_Solution_Samples_Stats.txt")
}
OUTPUT_FILE = Path("/home/zubair/Desk/A/features_final.tsv")

# --- OpenCRAVAT Features to Keep ---
OC_FEATURES = [
    'hugo_symbol', 'chromosome', 'start_position',
    'cadd.phred', 'revel.score', 'alphamissense.am_pathogenicity',
    'sift.score', 'polyphen2.hdiv_rank', 'gerp.gerp_rs'
]


def load_file(path, description):
    """Robust file loader with auto-format detection"""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description} file. Expected: {path}")

    try:
        return pd.read_csv(path, sep='\t' if path.suffix in ('.txt', '.tsv') else ',', low_memory=False)
    except Exception as e:
        raise ValueError(f"Error reading {path.name}: {str(e)}")


def calculate_exposures(prob_df):
    """Calculate signature exposures from probabilities"""
    print("\nCalculating exposures from probability file...")

    # Get all SBS columns
    sig_cols = [c for c in prob_df.columns if c.startswith('SBS')]
    if not sig_cols:
        raise ValueError("No SBS signature columns found in probability file")

    # Calculate mean probability per sample per signature
    exposures = prob_df.groupby('Sample Names')[sig_cols].mean()
    exposures = exposures.add_prefix('exp_').reset_index()

    print(f"Generated exposures for {len(exposures)} samples")
    return exposures


def main():
    print("=== Building Feature Table ===")

    # 1) Load all data
    print("\nLoading input files:")
    dfs = {}
    for name, path in INPUTS.items():
        dfs[name] = load_file(path, name)
        print(f"✅ {name}: {path.name} ({len(dfs[name])} rows)")

    # 2) Prepare mutation data
    mut = dfs["mutations"]
    req_mut_cols = ['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position',
                    'Reference_Allele', 'Tumor_Seq_Allele2']
    missing = [c for c in req_mut_cols if c not in mut.columns]
    if missing:
        raise ValueError(f"Mutation table missing columns: {missing}")

    # 3) Prepare SPA probabilities
    prob = dfs["spa_prob"]
    print(f"\nSPA Probability file columns: {prob.columns.tolist()[:10]}...")

    # Get all signature columns (SBS1, SBS2, etc.)
    sig_cols = [c for c in prob.columns if c.startswith('SBS')]
    print(f"Detected {len(sig_cols)} signature columns: {sig_cols[:5]}...")

    if not sig_cols:
        raise ValueError("No SBS signature columns found in probability file")

    # 4) Calculate exposures from probabilities
    exposures = calculate_exposures(prob)

    # 5) Prepare SPA stats (for additional metrics if needed)
    stats = dfs["spa_stats"]
    print(f"\nSPA Stats file columns: {stats.columns.tolist()}")

    # 6) Prepare OpenCRAVAT data
    oc = dfs["opencravat"]
    oc.columns = [c.lower().strip() for c in oc.columns]
    oc_avail = [极 for c in OC_FEATURES if c in oc.columns]
    print(f"\nAvailable OpenCRAVAT features: {oc_avail}")

    # 7) Merge everything
    print("\nMerging data...")

    # First merge mutations with SPA probabilities
    merged = mut.merge(
        prob,
        left_on=['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position',
                 'Reference_Allele', 'Tumor_Seq_Allele2'],
        right_on=['Sample Names', 'MutationType', 'MutationType', 'MutationType', 'MutationType'],
        how='left'
    )

    # Then merge with calculated exposures
    merged = merged.merge(
        exposures,
        left_on='Tumor_Sample_Barcode',
        right_on='Sample Names',
        how='left'
    )

    # Optional: merge with SPA stats
    merged = merged.merge(
        stats,
        left_on='Tumor_Sample_Barcode',
        right_on='Sample Names',
        how='left',
        suffixes=('', '_stats')
    )

    # Finally merge with OpenCRAVAT
    merged = merged.merge(
        oc[oc_avail],
        left_on=['Chromosome', 'Start_Position'],
        right_on=['chromosome', 'start_position'],
        how='left'
    )

    # 8) Cleanup
    merged = merged.drop(
        columns=['Sample Names', 'MutationType', 'chromosome', 'start_position', 'Sample Names_stats'],
        errors='ignore'
    )
    merged.to_csv(OUTPUT_FILE, sep='\t', index=False)

    print(f"\n✅ Success! Merged features saved to:\n{OUTPUT_FILE}")
    print(f"Final dimensions: {len(merged)} rows x {len(merged.columns)} columns")
    print("\nFirst few columns:")
    print(merged.columns.tolist()[:15])
    print("\nSample output:")
    print(merged.iloc[0:3, :5].to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nDEBUGGING HELP:")
        print("1. Verify these files exist:")
        for name, path in INPUTS.items():
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name}: {path}")
        print("\n2. Check file contents:")
        print(f"   head -n 3 {INPUTS['mutations']}")
        print(f"   head -n 3 {INPUTS['spa_prob']}")
        raise