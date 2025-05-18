import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

# Load data
df = pd.read_csv("mutations_with_signatures_and_DP.tsv", sep='\t')

# Optional: correct gene names
gene_corrections = {
    'Tm3': 'TP53', 'PTa1': 'PTEN', 'TTM': 'TTN', 'EBR1': 'EGFR',
    'MMC16': 'MUC16', 'NG': 'NOTCH1', 'RND2': 'KRAS', 'LRP2': 'LRP1B',
    'CSHn3': 'CSMD3', 'XRIP2': 'XIRP2'
}
df['Hugo_Symbol'] = df['Hugo_Symbol'].replace(gene_corrections)

# Split by cancer type
df_gbm = df[df['Cancer_Type'] == 'GBM']
df_luad = df[df['Cancer_Type'] == 'LUAD']

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
luad_top = get_top_genes(df_luad)

# Plotting top 10 mutated genes
def plot_top_genes(df, cancer_type, color):
    plt.figure(figsize=(8, 3))
    sns.barplot(x="Gene", y="Sample_Count", data=df, color=color, width=0.6)
    plt.title(f"Top 10 Mutated Genes in {cancer_type}", fontsize=12)
    plt.ylabel("Mutated Samples", fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

plot_top_genes(gbm_top, "GBM", "#87CEEB")
plot_top_genes(luad_top, "LUAD", "#D3D3D3")

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
common_genes = set(gbm_top['Gene']).intersection(set(luad_top['Gene']))
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
    luad_common = get_common_gene_counts(df_luad[df_luad['Hugo_Symbol'].isin(common_genes)], 'LUAD')
    combined_common = pd.concat([gbm_common, luad_common])

    plt.figure(figsize=(8, 3))
    sns.barplot(x='Gene', y='Sample_Count', hue='Cancer_Type', data=combined_common,
                palette=['#87CEEB', '#D3D3D3'], width=0.6)
    plt.title('Common Top Mutated Genes in GBM and LUAD', fontsize=12)
    plt.ylabel("Mutated Samples", fontsize=10)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(title="", fontsize=9)
    plt.tight_layout()
    plt.show()
    # Count of unique samples per cancer type
    sample_counts = df.groupby('Cancer_Type')['sample'].nunique().reset_index()
    sample_counts.columns = ['Cancer_Type', 'Unique_Sample_Count']
    print("🧾 Number of Unique Samples per Cancer Type:\n")
    print(sample_counts.to_string(index=False))

    comparison_results = []
    for gene in sorted(common_genes):
        or_val, p_val = fisher_enrichment(df_gbm, df_luad, gene)
        comparison_results.append({
            'Gene': gene,
            'OR (GBM vs LUAD)': round(or_val, 2),
            'p-value': round(p_val, 6)
        })

    results_df = pd.DataFrame(comparison_results).sort_values("p-value")
    print("\n🧾 Enrichment Test for Common Top Mutated Genes (GBM vs. LUAD)\n")
    print(results_df.to_string(index=False))
else:
    print("❗ No common top 10 genes between GBM and LUAD.")

# COSMIC signature analysis across cancer types
cosmic_signature_map = {
    'SBS1': 'Spontaneous deamination of 5-methylcytosine (Aging)',
    'SBS2': 'APOBEC activity',
    'SBS3': 'Defective HR DNA repair (BRCA1/2)',
    'SBS4': 'Tobacco smoking',
    'SBS5': 'Unknown',
    'SBS6': 'Mismatch repair deficiency',
    'SBS7a': 'UV light exposure',
    'SBS13': 'APOBEC activity',
    'SBS10a': 'POLE mutation',
    'SBS18': 'ROS damage',
    'SBS11': 'Temozolomide treatment',
    'SBS31': 'Platinum chemotherapy',
    'SBS35': 'Chemotherapy (e.g., cisplatin)',
    'SBS8': 'Unknown',
    'SBS17a': 'Oxidative damage or unknown'
}

cancer_types = df['Cancer_Type'].unique()
for cancer in cancer_types:
    data = df[df['Cancer_Type'] == cancer]
    print(f"\n====== {cancer} ======")

    # Top mutated genes
    top_genes = data.groupby('Hugo_Symbol')['sample'].nunique().sort_values(ascending=False).head(5)
    top_gene_list = top_genes.index.tolist()
    print(f"🔬 Top Mutated Genes: {', '.join(top_gene_list)}")

    # Top 3 mutation types
    top_mutations = data['Variant_Classification'].value_counts().head(3)
    print("📊 Top 3 Mutation Types:")
    for mut, count in top_mutations.items():
        print(f"   • {mut}: {count} mutations")

    # Mutation distribution in top genes
    mutation_distribution = (
        data[data['Hugo_Symbol'].isin(top_gene_list)]
        .groupby(['Hugo_Symbol', 'Variant_Classification'])
        .size()
        .unstack(fill_value=0)
    )
    print("\n🧬 Mutation Type Distribution in Top Genes:")
    print(mutation_distribution)

    # COSMIC signature contribution
    sbs_cols = [col for col in data.columns if col.startswith('SBS')]
    signature_mean_all = data[sbs_cols].mean().sort_values(ascending=False)
    top_signature_mean = signature_mean_all.head(10)
    signature_mapping = {sig: cosmic_signature_map.get(sig, 'Unknown') for sig in top_signature_mean.index}
    dominant_sig = top_signature_mean.index[0]
    print(f"\n🧪 Dominant COSMIC Signature: {dominant_sig} ({signature_mapping[dominant_sig]})")

    sig_df = pd.DataFrame({
        'Signature': top_signature_mean.index,
        'Mean_Contribution': top_signature_mean.values,
        'Etiology': [signature_mapping[sig] for sig in top_signature_mean.index]
    })
    print("\n📚 Top 10 Signature Contributions and Etiologies:")
    print(sig_df.to_string(index=False))
