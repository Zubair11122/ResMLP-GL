# Here's the full updated script integrating both fixes:
# 1) Survival-analysis drops NaNs and only writes non-empty PDFs.
# 2) Co-mutation heatmap uses os.path.join (no stray backslashes in filenames).

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import gseapy as gp
from matplotlib.backends.backend_pdf import PdfPages

# ── 0) Paths & output dirs ────────────────────────────────────────────────────
base_dir      = "C:/Users/Zubair/Desktop/B"
combined_file = os.path.join(base_dir, "mutations_with_clinical_combined.tsv")
plots_dir     = os.path.join(base_dir, "plots")
tables_dir    = os.path.join(base_dir, "tables")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# ── cosmic_signature_map ───────────────────────────────────────────────────────
cosmic_signature_map = {
    'SBS1': 'Spontaneous deamination of 5-methylcytosine (Aging)',
    'SBS2': 'APOBEC activity',
    'SBS3': 'Defective HR DNA repair (BRCA1/2)',
    'SBS4': 'Tobacco smoking',
    'SBS5': 'Unknown (clock-like)',
    'SBS6': 'Mismatch repair deficiency',
    'SBS7a': 'UV light exposure',
    'SBS7b': 'UV light exposure',
    'SBS7c': 'UV light exposure',
    'SBS7d': 'UV light exposure',
    'SBS8': 'Unknown',
    'SBS9': 'Polymerase eta activity',
    'SBS10a': 'POLE exonuclease domain mutation',
    'SBS10b': 'POLE exonuclease domain mutation',
    'SBS11': 'Temozolomide treatment',
    'SBS12': 'Unknown',
    'SBS13': 'APOBEC activity',
    'SBS14': 'POLE exonuclease domain mutation',
    'SBS15': 'Mismatch repair deficiency',
    'SBS16': 'Unknown',
    'SBS17a': 'Oxidative damage or unknown',
    'SBS17b': 'Oxidative damage or unknown',
    'SBS18': 'ROS damage',
    'SBS19': 'Unknown',
    'SBS20': 'Mismatch repair deficiency',
    'SBS21': 'Mismatch repair deficiency',
    'SBS22': 'Aristolochic acid exposure',
    'SBS23': 'Unknown',
    'SBS24': 'Aflatoxin exposure',
    'SBS25': 'Chemotherapy (alkylating agents)',
    'SBS26': 'Mismatch repair deficiency',
    'SBS27': 'Unknown',
    'SBS28': 'Mismatch repair deficiency',
    'SBS29': 'Tobacco chewing',
    'SBS30': 'Base excision repair deficiency',
    'SBS31': 'Platinum chemotherapy',
    'SBS32': 'Azathioprine treatment',
    'SBS33': 'Unknown',
    'SBS34': 'Unknown',
    'SBS35': 'Chemotherapy (e.g., cisplatin)',
    'SBS36': 'Defective base excision repair',
    'SBS37': 'Unknown',
    'SBS38': 'UV light exposure',
    'SBS39': 'Unknown',
    'SBS40': 'Unknown',
    'SBS41': 'Unknown',
    'SBS42': 'Haloalkane exposure',
    'SBS44': 'Defective mismatch repair',
    'SBS84': 'AID activity',
    'SBS85': 'Unknown',
    'SBS86': 'Unknown',
    'SBS87': 'Unknown',
    'SBS88': 'Unknown',
    'SBS89': 'Unknown',
    'SBS90': 'Unknown'
}

# ── 1) Load the combined mutations + clinical file ────────────────────────────
df = pd.read_csv(combined_file, sep="\t", dtype=str)

# ── 2) Optional: correct Hugo symbols ─────────────────────────────────────────
gene_corrections = {
    'Tm3': 'TP53', 'PTa1': 'PTEN', 'TTM': 'TTN', 'EBR1': 'EGFR',
    'MMC16': 'MUC16', 'NG': 'NOTCH1', 'RND2': 'KRAS', 'LRP2': 'LRP1B',
    'CSHn3': 'CSMD3', 'XRIP2': 'XIRP2'
}
df['Hugo_Symbol'] = df['Hugo_Symbol'].replace(gene_corrections)

# ── 3) Ensure numeric survival columns ─────────────────────────────────────────
df['Overall Survival (Months)'] = pd.to_numeric(df['Overall Survival (Months)'], errors='coerce')
df['Overall Survival Status']   = pd.to_numeric(df['Overall Survival Status'],   errors='coerce').astype('Int64')

# ── 4) Split by cancer type ────────────────────────────────────────────────────
df_gbm  = df[df['Cancer_Type'] == 'GBM'].copy()
df_luad = df[df['Cancer_Type'] == 'LUAD'].copy()

# ── 5) Diagnostic checks ───────────────────────────────────────────────────────
for cancer_df, name in [(df_gbm, 'GBM'), (df_luad, 'LUAD')]:
    total = cancer_df['sample'].nunique()
    with_surv = cancer_df.loc[cancer_df['Overall Survival Status'].notna(), 'sample'].nunique()
    print(f"{name}: {total} samples; {with_surv} with survival data")

# ── 6) Top mutated genes ───────────────────────────────────────────────────────
def get_top_genes(df, n=10):
    return (
        df.groupby("Hugo_Symbol")["sample"]
          .nunique()
          .sort_values(ascending=False)
          .head(n)
          .reset_index(name="Sample_Count")
          .rename(columns={"Hugo_Symbol":"Gene"})
    )

gbm_top  = get_top_genes(df_gbm, 10)
luad_top = get_top_genes(df_luad, 10)

def plot_top_genes(df_top, cancer_type, color, fname):
    plt.figure(figsize=(8,4))
    ax = sns.barplot(x='Gene', y='Sample_Count', data=df_top, color=color, width=0.6)
    plt.title(f"Top {len(df_top)} Mutated Genes in {cancer_type}", fontsize=12)
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x()+p.get_width()/2, h+1, str(h), ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    for ext in ('png','pdf'):
        plt.savefig(os.path.join(plots_dir, f"{fname}.{ext}"), dpi=900)
    plt.close()

plot_top_genes(gbm_top,  'GBM',  "#87CEEB", "gbm_top_genes")
plot_top_genes(luad_top, 'LUAD', "#D3D3D3", "luad_top_genes")

# ── 7) Fisher’s Exact on common top genes ──────────────────────────────────────
common = set(gbm_top['Gene']).intersection(luad_top['Gene'])
if common:
    results = []
    for gene in sorted(common):
        a = df_gbm[df_gbm['Hugo_Symbol']==gene]['sample'].nunique()
        b = df_luad[df_luad['Hugo_Symbol']==gene]['sample'].nunique()
        t1, t2 = df_gbm['sample'].nunique(), df_luad['sample'].nunique()
        orv, pv = fisher_exact([[a,b],[t1-a,t2-b]])
        results.append({'Gene':gene,'OR':round(orv,2),'p-value':pv})
    res_df = pd.DataFrame(results).sort_values('p-value')
    res_df.to_csv(os.path.join(tables_dir, "gene_enrichment_results.csv"), index=False)
    res_df.to_excel(os.path.join(tables_dir, "gene_enrichment_results.xlsx"), index=False)

    comb = []
    for name, d in [('GBM', df_gbm), ('LUAD', df_luad)]:
        tmp = (
            d[d['Hugo_Symbol'].isin(common)]
             .groupby('Hugo_Symbol')['sample']
             .nunique()
             .reset_index(name='Sample_Count')
             .assign(Cancer_Type=name)
             .rename(columns={'Hugo_Symbol':'Gene'})
        )
        comb.append(tmp)
    comb_df = pd.concat(comb, ignore_index=True)
    plt.figure(figsize=(8,4))
    sns.barplot(x='Gene', y='Sample_Count', hue='Cancer_Type',
                data=comb_df, palette=['#87CEEB','#D3D3D3'], width=0.6)
    plt.title("Common Top Mutated Genes in GBM and LUAD", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    for ext in ('png','pdf'):
        plt.savefig(os.path.join(plots_dir, f"common_genes_comparison.{ext}"), dpi=900)
    plt.close()
else:
    print("❗ No common top 10 genes")

# ── 8) Signature heatmaps ──────────────────────────────────────────────────────
def plot_signature_heatmap(df, cancer_type):
    sbs_cols = sorted([c for c in df.columns if c.startswith('SBS')])
    means = df[sbs_cols].astype(float).mean().sort_values(ascending=False).head(15)
    sig_df = pd.DataFrame({
        'Signature': means.index,
        'Mean_Contribution': means.values,
        'Etiology': [cosmic_signature_map.get(sig,'Unknown') for sig in means.index]
    })
    sig_df.to_csv(os.path.join(tables_dir, f"{cancer_type}_signature_contributions.csv"), index=False)
    sig_df.to_excel(os.path.join(tables_dir, f"{cancer_type}_signature_contributions.xlsx"), index=False)
    plt.figure(figsize=(12,6))
    sns.heatmap(means.to_frame().T, annot=True, fmt=".2f",
                cbar_kws={'label':'Mean Contribution'})
    plt.title(f"Top 15 COSMIC Signatures in {cancer_type}", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    for ext in ('png','pdf'):
        plt.savefig(os.path.join(plots_dir, f"{cancer_type}_signature_heatmap.{ext}"), dpi=900)
    plt.close()

plot_signature_heatmap(df_gbm,  'GBM')
plot_signature_heatmap(df_luad, 'LUAD')

# ── 9) Co-mutation patterns ───────────────────────────────────────────────────
def analyze_co_mutation(df, cancer_type, top_n=5):
    top = get_top_genes(df, top_n)['Gene'].tolist()
    samp = df['sample'].unique()
    mat = pd.DataFrame(index=samp)
    for g in top:
        mat[g] = mat.index.isin(df[df['Hugo_Symbol']==g]['sample']).astype(int)

    co = pd.DataFrame(index=top, columns=top, dtype=float)
    for g1 in top:
        for g2 in top:
            if g1 == g2:
                co.loc[g1, g2] = 1.0
            else:
                both   = ((mat[g1] & mat[g2]) == 1).sum()
                either = ((mat[g1] | mat[g2]) == 1).sum()
                co.loc[g1, g2] = both / either if either > 0 else 0

    plt.figure(figsize=(8,6))
    sns.heatmap(co, annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Co-mutation in {cancer_type}", fontsize=12)
    plt.tight_layout()

    for ext in ('png','pdf'):
        fname = os.path.join(plots_dir, f"{cancer_type}_comutation_heatmap.{ext}")
        plt.savefig(fname, dpi=900)
    plt.close()

    return co

gbm_co  = analyze_co_mutation(df_gbm,  'GBM')
luad_co = analyze_co_mutation(df_luad, 'LUAD')

# ── 10) Mutational burden ───────────────────────────────────────────────────────
def compare_mutational_burden(df1, df2, n1, n2):
    b1 = df1.groupby('sample').size().reset_index(name='mutation_count')
    b2 = df2.groupby('sample').size().reset_index(name='mutation_count')
    b1['Cancer_Type'], b2['Cancer_Type'] = n1, n2
    comb = pd.concat([b1, b2], ignore_index=True)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Cancer_Type', y='mutation_count', data=comb)
    plt.title("Mutational Burden Comparison", fontsize=12)
    plt.tight_layout()
    for ext in ('png','pdf'):
        plt.savefig(os.path.join(plots_dir, f"mutational_burden_comparison.{ext}"), dpi=900)
    plt.close()

    stat, p = mannwhitneyu(b1['mutation_count'], b2['mutation_count'])
    print(f"Mann-Whitney U {n1} vs {n2}: U={stat:.1f}, p={p:.4f}")

    stats = pd.DataFrame({
        'Cancer_Type':     [n1,          n2],
        'Median_Mutations':[b1['mutation_count'].median(), b2['mutation_count'].median()],
        'Mean_Mutations':  [b1['mutation_count'].mean(),  b2['mutation_count'].mean()],
        'Mann_Whitney_p':  [np.nan,      p]
    })
    stats.to_csv(os.path.join(tables_dir, "mutational_burden_stats.csv"), index=False)
    stats.to_excel(os.path.join(tables_dir, "mutational_burden_stats.xlsx"), index=False)
    return comb

burden = compare_mutational_burden(df_gbm, df_luad, 'GBM', 'LUAD')

# ── 11) Cancer‐specific summaries ───────────────────────────────────────────────
def analyze_cancer_type(df, cancer_type):
    print(f"\n====== {cancer_type} ======")
    top5 = get_top_genes(df, 5)
    print("Top 5 Genes:", ", ".join(top5['Gene']))
    vc = df['Variant_Classification'].value_counts().head(3)
    print("Top 3 Mutation Types:")
    for m, c in vc.items():
        print(f" • {m}: {c}")

    sbs   = sorted([c for c in df.columns if c.startswith('SBS')])
    means = df[sbs].astype(float).mean().sort_values(ascending=False)
    dom   = means.index[0]
    print(f"\nDominant COSMIC Signature: {dom} ({cosmic_signature_map.get(dom)})")

    top10 = means.head(10)
    etio  = [cosmic_signature_map.get(sig,'Unknown') for sig in top10.index]
    out   = pd.DataFrame({
        'Signature':        top10.index,
        'Mean_Contribution':[v for v in top10.values],
        'Etiology':         etio
    })
    print("\nTop 10 Signature Contributions & Etiologies:")
    print(out.to_string(index=False))

analyze_cancer_type(df_gbm, 'GBM')
analyze_cancer_type(df_luad, 'LUAD')

# ── 12) Survival analysis ──────────────────────────────────────────────────────
def perform_survival_analysis(df, cancer_type, top_n=5):
    clinical = (
        df[['sample','Overall Survival (Months)','Overall Survival Status']]
        .drop_duplicates('sample')
        .dropna(subset=['Overall Survival (Months)','Overall Survival Status'])
        .reset_index(drop=True)
    )

    top_genes = get_top_genes(df, top_n)['Gene']
    ok = []
    for gene in top_genes:
        muts = set(df[df['Hugo_Symbol']==gene]['sample'])
        surv = clinical.assign(mutated=clinical['sample'].isin(muts).astype(int))
        if surv['mutated'].sum() >= 3 and (surv['mutated']==0).sum() >= 3:
            ok.append(gene)
        else:
            print(f"⚠️ Skipping {gene} in {cancer_type} (not enough events)")

    if not ok:
        print(f"No genes passed filtering for survival in {cancer_type}")
        return

    pdf_path = os.path.join(plots_dir, f"{cancer_type}_survival_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        for gene in ok:
            surv = clinical.assign(mutated=clinical['sample'].isin(
                df[df['Hugo_Symbol']==gene]['sample']
            ).astype(int))
            surv = surv.dropna(subset=['Overall Survival (Months)','Overall Survival Status'])

            kmf = KaplanMeierFitter()
            plt.figure(figsize=(8,6))

            kmf.fit(
                surv.loc[surv['mutated']==1, 'Overall Survival (Months)'],
                surv.loc[surv['mutated']==1, 'Overall Survival Status'],
                label=f"{gene} Mut"
            ).plot(ci_show=True)

            kmf.fit(
                surv.loc[surv['mutated']==0, 'Overall Survival (Months)'],
                surv.loc[surv['mutated']==0, 'Overall Survival Status'],
                label=f"{gene} WT"
            ).plot(ax=plt.gca(), ci_show=True)

            res = logrank_test(
                surv.loc[surv['mutated']==1, 'Overall Survival (Months)'],
                surv.loc[surv['mutated']==0, 'Overall Survival (Months)'],
                surv.loc[surv['mutated']==1, 'Overall Survival Status'],
                surv.loc[surv['mutated']==0, 'Overall Survival Status']
            )

            plt.title(f"{gene} survival in {cancer_type} (p={res.p_value:.4f})")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"✅ Survival curves saved to {pdf_path}")

perform_survival_analysis(df_gbm, 'GBM')
perform_survival_analysis(df_luad, 'LUAD')

# ── 13) Optional Pathway enrichment ────────────────────────────────────────────
try:
    def run_pathway(df, cancer_type, top_n=10):
        genes = get_top_genes(df, top_n)['Gene'].tolist()
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=['KEGG_2021_Human','Reactome_2022','GO_Biological_Process_2023'],
            organism='human'
        )
        enr.results.to_csv(os.path.join(tables_dir, f"{cancer_type}_pathway_full.csv"), index=False)
        enr.results.to_excel(os.path.join(tables_dir, f"{cancer_type}_pathway_full.xlsx"), index=False)

        for db in enr.results['Gene_set'].unique():
            plt.figure(figsize=(8,4))
            gp.dotplot(
                enr.results[enr.results['Gene_set']==db],
                column='Adjusted P-value',
                x='Gene_set',
                title=f"{cancer_type} {db}",
                figsize=(8,4)
            )
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{cancer_type}_{db}_pathway.png"), dpi=900)
            plt.close()

    run_pathway(df_gbm, 'GBM')
    run_pathway(df_luad, 'LUAD')

except ImportError:
    print("⚠️ gseapy missing — skipping pathway")
except Exception as e:
    print("⚠️ Pathway analysis failed:", e)

print("\n✅ All analysis complete — check 'plots/' and 'tables/'")

