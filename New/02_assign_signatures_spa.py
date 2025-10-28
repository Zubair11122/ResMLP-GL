#!/usr/bin/env python3
# 02_assign_signatures_spa.py - CUSTOM FILENAME VERSION
from SigProfilerAssignment import Analyzer as spa
import os
import pandas as pd

# ---------- USER SETTINGS ----------
MATRIX_FILE = "/home/zubair/Desk/AB/Final-project-files/spm_out_coad/SBS/TCGA_COAD.SBS96.exome"  # Your exact file path
SIGNATURES_FILE = "/home/zubair/Desk/AB/Final-project-files/allowed_SBS.tsv"
OUTPUT_DIR = "/home/zubair/Desk/AB/Final-project-files/spa_out_restricted_coad"
GENOME_BUILD = "GRCh38"


# -----------------------------------

def verify_and_convert_matrix():
    """Handle custom matrix filename and format"""
    print(f"\nChecking matrix file: {MATRIX_FILE}")

    if not os.path.exists(MATRIX_FILE):
        raise FileNotFoundError(f"Matrix file not found at {MATRIX_FILE}")

    # Convert to standard format if needed
    standard_path = os.path.join(os.path.dirname(MATRIX_FILE), "SBS96.all")

    if not os.path.exists(standard_path):
        print("Converting custom matrix to standard format...")
        try:
            df = pd.read_csv(MATRIX_FILE, sep='\t')
            df.to_csv(standard_path, sep='\t', index=False)
            print(f"✅ Created standard format matrix at: {standard_path}")
        except Exception as e:
            raise ValueError(f"Failed to convert matrix file: {str(e)}")

    return standard_path


def verify_signatures():
    """Verify signatures file exists"""
    if not os.path.exists(SIGNATURES_FILE):
        raise FileNotFoundError(f"Signatures file not found at {SIGNATURES_FILE}")
    print(f"✅ Signatures file found: {SIGNATURES_FILE}")


def main():
    print("=== Signature Assignment Analysis ===")

    # 1) Verify and prepare inputs
    processed_matrix = verify_and_convert_matrix()
    verify_signatures()

    # 2) Clean output directory
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 3) Run analysis
    print(f"\nStarting analysis with:")
    print(f"- Matrix: {processed_matrix}")
    print(f"- Signatures: {SIGNATURES_FILE}")
    print(f"- Output: {OUTPUT_DIR}")

    try:
        spa.cosmic_fit(
            samples=processed_matrix,
            output=OUTPUT_DIR,
            signatures=SIGNATURES_FILE,
            exome=True,
            genome_build=GENOME_BUILD,
            make_plots=True,
            export_probabilities=True,
            export_probabilities_per_mutation=True,
            verbose=True
        )
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Verify matrix file format:")
        print(f"   head -n 5 {processed_matrix}")
        print("2. Check signatures format:")
        print(f"   head -n 5 {SIGNATURES_FILE}")
        print("3. Try manual run:")
        print(f"   python -m SigProfilerAssignment {processed_matrix} {OUTPUT_DIR} \\")
        print(f"   --signatures {SIGNATURES_FILE} --exome --genome GRCh38 --verbose")
        raise

    # 4) Verify outputs
    print("\n✅ Analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    os.system(f"tree {OUTPUT_DIR} | head -15")


if __name__ == "__main__":
    main()