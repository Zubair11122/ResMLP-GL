#!/usr/bin/env python3
# 01_make_matrices_exome.py - FINAL WORKING VERSION
import os
import pandas as pd
from pathlib import Path
from SigProfilerMatrixGenerator import install as genInstall
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc

# ---------- USER SETTINGS ----------
SOURCE_FILE = "/home/zubair/Desk/AB/Final-project-files/TCGA-COAD.somaticmutation_wxs.tsv"
GENOME_BUILD = "GRCh38"  # Must be exactly "GRCh38" or "GRCh37"
PROJECT_NAME = "TCGA_COAD"
OUT_DIR = "/home/zubair/Desk/AB/Final-project-files/spm_out_coad"
INPUT_DIR = "/home/zubair/Desk/AB/Final-project-files/input_variants_coad"
COMBINED_MAF = os.path.join(INPUT_DIR, "combined.maf")


# -----------------------------------

def effect_to_variant_classification(eff: str) -> str:
    """Convert effect string to MAF variant classification."""
    if not isinstance(eff, str):
        return "Unknown"
    e = eff.lower()
    if "missense" in e:               return "Missense_Mutation"
    if "synonymous" in e:             return "Silent"
    if "stop_gained" in e or "nonsense" in e: return "Nonsense_Mutation"
    if "stop_lost" in e:              return "Nonstop_Mutation"
    if "splice" in e:                 return "Splice_Site"
    if "frameshift" in e and "ins" in e: return "Frame_Shift_Ins"
    if "frameshift" in e and "del" in e: return "Frame_Shift_Del"
    if "frameshift" in e:             return "Frame_Shift_Ins"
    if "inframe" in e and "ins" in e: return "In_Frame_Ins"
    if "inframe" in e and "del" in e: return "In_Frame_Del"
    if "utr" in e:                    return "5'UTR" if "5" in e else "3'UTR"
    if "intronic" in e or "intron" in e: return "Intron"
    return "Unknown"


def infer_variant_type(ref: str, alt: str) -> str:
    """Determine variant type from ref/alt alleles."""
    if pd.isna(ref) or pd.isna(alt):
        return "UNKNOWN"

    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()

    if not ref or not alt:
        return "UNKNOWN"

    if len(ref) == 1 and len(alt) == 1:
        return "SNP"
    if len(ref) == len(alt):
        return "DNP" if len(ref) == 2 else "ONP"
    if len(alt) > len(ref):
        return "INS"
    if len(alt) < len(ref):
        return "DEL"
    return "UNKNOWN"


def main():
    # 0) Prepare folders
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Clean output directory (critical for reliable runs)
    if os.path.exists(OUT_DIR):
        import shutil
        shutil.rmtree(OUT_DIR)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Install genome reference if needed
    print(f"Checking genome reference installation for {GENOME_BUILD}...")
    try:
        genInstall.install(GENOME_BUILD)
        print(f"✅ Genome reference {GENOME_BUILD} is installed")
    except Exception as e:
        print(f"❌ Genome installation failed: {str(e)}")
        raise

    # 2) Load and clean input data
    print(f"Loading input file: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE, sep="\t", dtype=str)

    # Convert NA/NaN to empty strings in key columns
    for col in ['ref', 'alt', 'gene', 'chrom', 'start', 'end', 'effect']:
        df[col] = df[col].fillna('')

    # Ensure expected columns exist
    for col in ["sample", "gene", "chrom", "start", "end", "ref", "alt",
                "Tumor_Sample_Barcode", "Amino_Acid_Change", "effect"]:
        if col not in df.columns:
            df[col] = ""

    # 3) Build MAF dataframe
    print("Converting to MAF format...")
    maf = pd.DataFrame()
    maf["Hugo_Symbol"] = df["gene"]
    maf["Entrez_Gene_Id"] = ""
    maf["Center"] = ""
    maf["NCBI_Build"] = GENOME_BUILD
    maf["Chromosome"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    maf["Start_Position"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    maf["End_Position"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    maf["Strand"] = "+"
    maf["Variant_Classification"] = df["effect"].apply(effect_to_variant_classification)
    maf["Variant_Type"] = [infer_variant_type(r, a) for r, a in zip(df["ref"], df["alt"])]
    maf["Reference_Allele"] = df["ref"].str.upper()
    maf["Tumor_Seq_Allele1"] = ""
    maf["Tumor_Seq_Allele2"] = df["alt"].str.upper()
    maf["dbSNP_RS"] = ""
    maf["dbSNP_Val_Status"] = ""
    maf["Tumor_Sample_Barcode"] = df["Tumor_Sample_Barcode"]
    maf["Matched_Norm_Sample_Barcode"] = ""
    maf["Match_Norm_Seq_Allele1"] = ""
    maf["Match_Norm_Seq_Allele2"] = ""
    maf["Tumor_Validation_Allele1"] = ""
    maf["Tumor_Validation_Allele2"] = ""
    maf["Match_Norm_Validation_Allele1"] = ""
    maf["Match_Norm_Validation_Allele2"] = ""
    maf["Verification_Status"] = ""
    maf["Validation_Status"] = ""
    maf["Mutation_Status"] = "Somatic"
    maf["Sequencing_Phase"] = ""
    maf["Sequence_Source"] = "WXS"
    maf["Validation_Method"] = ""
    maf["Score"] = ""
    maf["BAM_File"] = ""
    maf["Sequencer"] = ""
    maf["Tumor_Sample_UUID"] = ""
    maf["Matched_Norm_Sample_UUID"] = ""
    maf["Protein_Change"] = df["Amino_Acid_Change"]

    # 4) Filter invalid rows
    critical = ["Chromosome", "Start_Position", "End_Position",
                "Reference_Allele", "Tumor_Seq_Allele2", "Tumor_Sample_Barcode"]
    before = len(maf)
    maf = maf.dropna(subset=critical)
    maf = maf[(maf["Reference_Allele"] != "") &
              (maf["Tumor_Seq_Allele2"] != "") &
              (maf["Tumor_Sample_Barcode"] != "")]
    after = len(maf)

    print(f"Prepared MAF rows: kept {after} / {before}")

    # 5) Write combined MAF
    maf.to_csv(COMBINED_MAF, sep="\t", index=False)
    print(f"✅ Wrote combined MAF: {COMBINED_MAF}")

    # 6) Generate matrices - ROBUST VERSION
    print("Generating mutation matrices...")

    try:
        # VERBOSE DEBUG OUTPUT
        print("\nRunning matrix generator with:")
        print(f"Project: {PROJECT_NAME}")
        print(f"Genome: {GENOME_BUILD}")
        print(f"Input dir: {INPUT_DIR} (contains: {os.listdir(INPUT_DIR)})")
        print(f"Output dir: {OUT_DIR}")

        # THE WORKING FUNCTION CALL
        matrices = SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc(
            project=PROJECT_NAME,
            reference_genome=GENOME_BUILD,
            path_to_input_files=INPUT_DIR,
            exome=True,
            output_directory=OUT_DIR,  # Explicit output directory
            chrom_based=False,
            plot=False,
            tsb_stat=False,
            seqInfo=False
        )

        # ROBUST OUTPUT VERIFICATION
        print("\nChecking output directory...")
        found_files = []
        for root, dirs, files in os.walk(OUT_DIR):
            for file in files:
                if file.endswith(('.all', '.txt', '.csv')):
                    found_files.append(os.path.join(root, file))

        if found_files:
            print("✅ Success! Found these matrix files:")
            for f in found_files:
                print(f"- {f}")

            # Specifically check for SBS96
            sbs96_path = os.path.join(OUT_DIR, "SBS", "96", "matrices", "SBS96.all")
            if os.path.exists(sbs96_path):
                print(f"\nMain SBS96 matrix at: {sbs96_path}")
                print("First 5 lines:")
                os.system(f"head -n 5 {sbs96_path}")
            else:
                print("\nℹ️ SBS96.all not found, but other matrices exist")
        else:
            raise FileNotFoundError(f"No matrix files found in {OUT_DIR}")

    except Exception as e:
        print(f"\n❌ Matrix generation failed: {str(e)}")
        print("\nADVANCED TROUBLESHOOTING:")
        print("1. Verify MAF file content:")
        print(f"   head -n 10 {COMBINED_MAF}")
        print("2. Check disk space:")
        print("   df -h /home/zubair/Desk/A/")
        print("3. Try manual generation:")
        print(
            f"   python -m SigProfilerMatrixGenerator {PROJECT_NAME} {GENOME_BUILD} {INPUT_DIR} --exome --output {OUT_DIR} --verbose")
        print("4. Check package integrity:")
        print(
            "   python -c \"from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc; print(dir(SigProfilerMatrixGeneratorFunc))\"")
        raise


if __name__ == "__main__":
    main()
    