import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

# Create output directory for figures, if it doesn't exist
fig_dir = "C:/Users/Zubair/Desktop/AB/Final-project-files/figures"
os.makedirs(fig_dir, exist_ok=True)

# Load data
df = pd.read_csv("C:/Users/Zubair/Desktop/AB/Final-project-files/mutations_with_signatures_and_DP.tsv", sep='\t')

# Optional: correct gene names
gene_corrections = {
    'Tm3': 'TP53', 'PTa1': 'PTEN', 'TTM': 'TTN', 'EBR1': 'EGFR',
    'MMC16': 'MUC16', 'NG': 'NOTCH1', 'RND2': 'KRAS', 'LRP2': 'LRP1B',
    'CSHn3': 'CSMD3', 'XRIP2': 'XIRP2'
}
df['Hugo_Symbol'] = df['Hugo_Symbol'].replace(gene_corrections)

# Split by cancer type
df_gbm = df[df['Cancer_Type'] == 'GBM']
df_COAD = df[df['Cancer_Type'] == 'COAD']

# Top mutated genes
def get_top_genes(df):
    return (
        df.groupby("Hugo_Symbol")["sample"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"Hugo_Symbol": "Gene", "sample": "Sample_Count"})
    )

gbm_top = get_top_genes(df_gbm)
COAD_top = get_top_genes(df_COAD)

# Plotting top 10 mutated genes and saving figures
def plot_top_genes(df, cancer_type, color):
    plt.figure(figsize=(8, 3))
    sns.barplot(x="Gene", y="Sample_Count", data=df, color=color, width=0.6)
    plt.title(f"Top 10 Mutated Genes in {cancer_type}", fontsize=12)
    plt.ylabel("Mutated Samples", fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    # Save figure
    fname = os.path.join(fig_dir, f"top10_mutated_genes_{cancer_type}.png")
    plt.savefig(fname, dpi=900)
    print(f"Saved: {fname}")
    plt.show()
    plt.close()

plot_top_genes(gbm_top, "GBM", "#87CEEB")
plot_top_genes(COAD_top, "COAD", "#D3D3D3")

# Fisher's Exact Test
def fisher_enrichment(df1, df2, gene):
    a = df1[df1['Hugo_Symbol'] == gene]['sample'].nunique()
    b = df2[df2['Hugo_Symbol'] == gene]['sample'].nunique()
    total1 = df1['sample'].nunique()
    total2 = df2['sample'].nunique()
    table = [[a, b], [total1 - a, total2 - b]]
    odds_ratio, p_value = fisher_exact(table)
    return odds_ratio, p_value

# Compare common top mutated genes
common_genes = set(gbm_top['Gene']).intersection(set(COAD_top['Gene']))
if common_genes:
    def get_common_gene_counts(df, cancer_type):
        return (
            df.groupby('Hugo_Symbol')['sample']
            .nunique()
            .reset_index()
            .rename(columns={'Hugo_Symbol': 'Gene', 'sample': 'Sample_Count'})
            .assign(Cancer_Type=cancer_type)
        )
    gbm_common = get_common_gene_counts(df_gbm[df_gbm['Hugo_Symbol'].isin(common_genes)], 'GBM')
    COAD_common = get_common_gene_counts(df_COAD[df_COAD['Hugo_Symbol'].isin(common_genes)], 'COAD')
    combined_common = pd.concat([gbm_common, COAD_common])

    plt.figure(figsize=(8, 3))
    sns.barplot(
        x='Gene', y='Sample_Count', hue='Cancer_Type',
        data=combined_common, palette=['#87CEEB', '#D3D3D3'], width=0.6
    )
    plt.title('Common Top Mutated Genes in GBM and COAD', fontsize=12)
    plt.ylabel("Mutated Samples", fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(title="", fontsize=9)
    plt.tight_layout()
    # Save combined plot
    fname = os.path.join(fig_dir, "common_top_mutated_genes_GBM_vs_COAD.png")
    plt.savefig(fname, dpi=900)
    print(f"Saved: {fname}")
    plt.show()
    plt.close()

    # Count of unique samples per cancer type
    sample_counts = df.groupby('Cancer_Type')['sample'].nunique().reset_index()
    sample_counts.columns = ['Cancer_Type', 'Unique_Sample_Count']
    print("🧾 Number of Unique Samples per Cancer Type:\n")
    print(sample_counts.to_string(index=False))

    comparison_results = []
    for gene in sorted(common_genes):
        or_val, p_val = fisher_enrichment(df_gbm, df_COAD, gene)
        comparison_results.append({
            'Gene': gene,
            'OR (GBM vs COAD)': round(or_val, 2),
            'p-value': round(p_val, 6)
        })

    results_df = pd.DataFrame(comparison_results).sort_values("p-value")
    print("\n🧾 Enrichment Test for Common Top Mutated Genes (GBM vs. COAD)\n")
    print(results_df.to_string(index=False))
else:
    print("❗ No common top 10 genes between GBM and COAD.")

# COSMIC signature analysis across cancer types (prints only)
cosmic_signature_map = {
    # ─── Core COSMIC v3.4 SBS Signatures (Complete & Corrected) ────────────
    'SBS1': 'Spontaneous deamination of 5-methylcytosine (clock-like, aging)',
    'SBS2': 'APOBEC cytidine deaminase activity (predominantly SBS2)',
    'SBS3': 'Defective homologous recombination repair (BRCA1/2 mutations)',
    'SBS4': 'Tobacco mutagens (smoking-associated)',
    'SBS5': 'Clock-like mutational process (unknown etiology)',
    'SBS6': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS7a': 'Ultraviolet light exposure (melanoma, early replication)',
    'SBS7b': 'Ultraviolet light exposure (melanoma, late replication)',
    'SBS7c': 'Ultraviolet light exposure (melanoma, transcription-coupled)',
    'SBS7d': 'Ultraviolet light exposure (melanoma, unknown strand bias)',
    'SBS8': 'Unknown etiology (associated with lymphoid malignancies)',
    'SBS9': 'Polymerase eta somatic hypermutation activity',
    'SBS10a': 'POLE exonuclease domain mutations (ultramutated tumors)',
    'SBS10b': 'POLE exonuclease domain mutations (ultramutated tumors)',
    'SBS11': 'Temozolomide chemotherapy treatment',
    'SBS12': 'Unknown etiology (liver cancer-associated)',
    'SBS13': 'APOBEC cytidine deaminase activity (predominantly SBS13)',
    'SBS14': 'POLE exonuclease domain mutations (ultramutated tumors)',
    'SBS15': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS16': 'Unknown etiology (stomach cancer-associated)',
    'SBS17a': 'Oxidative damage (unknown source)',
    'SBS17b': 'Oxidative damage (unknown source)',
    'SBS18': 'Reactive oxygen species (ROS) damage',
    'SBS19': 'Unknown etiology (pilocytic astrocytoma-associated)',
    'SBS20': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS21': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS22': 'Aristolochic acid exposure (urothelial cancers)',
    'SBS23': 'Unknown etiology',
    'SBS24': 'Aflatoxin exposure (liver cancers)',
    'SBS25': 'Chemotherapy with alkylating agents',
    'SBS26': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS27': 'Unknown etiology',
    'SBS28': 'Defective DNA mismatch repair (microsatellite unstable tumors)',
    'SBS29': 'Tobacco chewing (betel quid-associated)',
    'SBS30': 'Defective base excision repair (NTHL1 mutations)',
    'SBS31': 'Platinum-based chemotherapy treatment',
    'SBS32': 'Azathioprine immunosuppressant treatment',
    'SBS33': 'Unknown etiology',
    'SBS34': 'Unknown etiology',
    'SBS35': 'Platinum-based chemotherapy treatment',
    'SBS36': 'Defective base excision repair (MUTYH mutations)',
    'SBS37': 'Unknown etiology',
    'SBS38': 'Ultraviolet light exposure (unknown strand bias)',
    'SBS39': 'Unknown etiology',
    'SBS40': 'Unknown etiology (lymphoid malignancy-associated)',
    'SBS41': 'Unknown etiology',
    'SBS42': 'Haloalkane exposure (paint industry chemicals)',
    'SBS43': 'Unknown etiology',
    'SBS44': 'Defective DNA mismatch repair (microsatellite unstable tumors)',

    # ─── Rare/Newly Discovered SBS Signatures (45-60) ─────────────────────
    'SBS45': 'Unknown etiology',
    'SBS46': 'Unknown etiology',
    'SBS47': 'Unknown etiology',
    'SBS48': 'Unknown etiology',
    'SBS49': 'Unknown etiology',
    'SBS50': 'Unknown etiology',
    'SBS51': 'Unknown etiology',
    'SBS52': 'Unknown etiology',
    'SBS53': 'Unknown etiology',
    'SBS54': 'Unknown etiology',
    'SBS55': 'Unknown etiology',
    'SBS56': 'Unknown etiology',
    'SBS57': 'Unknown etiology',
    'SBS58': 'Unknown etiology',
    'SBS59': 'Unknown etiology',
    'SBS60': 'Unknown etiology',

    # ─── Special Context Signatures (84+) ─────────────────────────────────
    'SBS84': 'Activation-induced cytidine deaminase (AID) activity',
    'SBS85': 'Unknown etiology (lymphoid malignancy-associated)',
    'SBS86': 'POLE exonuclease domain mutations (ultramutated tumors)',
    'SBS87': 'Unknown etiology',
    'SBS88': 'Colibactin exposure (pks+ E. coli)',
    'SBS89': 'Unknown etiology',
    'SBS90': 'Duocarmycin exposure (chemotherapy agent)',

    # ─── Newest Signatures (91-99) ───────────────────────────────────────
    'SBS91': 'Unknown etiology',
    'SBS92': 'Unknown etiology',
    'SBS93': 'Unknown etiology',
    'SBS94': 'Unknown etiology',
    'SBS95': 'Unknown etiology',
    'SBS96': 'Unknown etiology',
    'SBS97': 'Unknown etiology',
    'SBS98': 'Unknown etiology',
    'SBS99': 'Unknown etiology',

    # ─── Retired Signatures (Documentation Only) ──────────────────────────
    'SBS7': 'Legacy UV signature (now split into SBS7a-d)',
    'SBS10': 'Legacy POLE signature (now split into SBS10a/b)',
    'SBS17': 'Legacy oxidative damage signature (now split into SBS17a/b)'
}

for cancer in df['Cancer_Type'].unique():
    data = df[df['Cancer_Type'] == cancer]
    print(f"\n====== {cancer} ======")
    # ... (rest of print-based analysis)
