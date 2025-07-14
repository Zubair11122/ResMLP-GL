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
base_dir      = "C:/Users/Zubair/Desktop/AB/Final-project-files"
combined_file = os.path.join(base_dir, "mutations_with_clinical_combined.tsv")
plots_dir     = os.path.join(base_dir, "plots")
tables_dir    = os.path.join(base_dir, "tables")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# ── cosmic_signature_map ───────────────────────────────────────────────────────
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
# ── 1) Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(combined_file, sep="\t", dtype=str)

# ── Ensure correct sample identifier column ────────────────────────────────────
sample_col = None
for col in ['sample', 'Sample ID', 'Tumor_Sample_Barcode']:
    if col in df.columns:
        sample_col = col
        break
if sample_col is None:
    raise KeyError("No column found for sample identifier! Checked: 'sample', 'Sample ID', 'Tumor_Sample_Barcode'.")
df = df.rename(columns={sample_col: 'sample'})

# ── 2) Hugos ───────────────────────────────────────────────────────────────────
gene_corrections = {
    'Tm3': 'TP53', 'PTa1': 'PTEN', 'TTM': 'TTN', 'EBR1': 'EGFR',
    'MMC16': 'MUC16', 'NG': 'NOTCH1', 'RND2': 'KRAS', 'LRP2': 'LRP1B',
    'CSHn3': 'CSMD3', 'XRIP2': 'XIRP2'
}
df['Hugo_Symbol'] = df['Hugo_Symbol'].replace(gene_corrections)

# ── 3) Survival numeric ────────────────────────────────────────────────────────
df['Overall Survival (Months)'] = pd.to_numeric(df['Overall Survival (Months)'], errors='coerce')
df['Overall Survival Status']   = pd.to_numeric(df['Overall Survival Status'],   errors='coerce').astype('Int64')

# ── 4) Split types ─────────────────────────────────────────────────────────────
df_gbm  = df[df['Cancer_Type'] == 'GBM'].copy()
df_COAD = df[df['Cancer_Type'] == 'COAD'].copy()

# ── 5) Diagnostics ─────────────────────────────────────────────────────────────
for cancer_df, name in [(df_gbm, 'GBM'), (df_COAD, 'COAD')]:
    total = cancer_df['sample'].nunique()
    with_surv = cancer_df.loc[cancer_df['Overall Survival Status'].notna(), 'sample'].nunique()
    print(f"{name}: {total} samples; {with_surv} with survival data")

# ── 6) Top genes ────────────────────────────────────────────────────────────────
def get_top_genes(df, n=10):
    return (df.groupby("Hugo_Symbol")["sample"]
              .nunique()
              .sort_values(ascending=False)
              .head(n)
              .reset_index(name="Sample_Count")
              .rename(columns={"Hugo_Symbol":"Gene"}))

gbm_top  = get_top_genes(df_gbm, 10)
COAD_top = get_top_genes(df_COAD, 10)

# ── 7) Plot top genes ──────────────────────────────────────────────────────────
def plot_top_genes(df_top, cancer_type, color, fname):
    plt.figure(figsize=(8,4))
    ax = sns.barplot(x='Gene', y='Sample_Count', data=df_top, color=color, width=0.6)
    plt.title(f"Top {len(df_top)} Mutated Genes in {cancer_type}", fontsize=12)
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2, p.get_height()+1, str(int(p.get_height())),
                ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(plots_dir, f"{fname}.{ext}"), dpi=900)
    plt.close()

plot_top_genes(gbm_top, 'GBM',  "#87CEEB", "gbm_top_genes")
plot_top_genes(COAD_top,'COAD',"#D3D3D3","COAD_top_genes")

# ── 8) Fisher exact common genes ───────────────────────────────────────────────
common = set(gbm_top['Gene']).intersection(COAD_top['Gene'])
if common:
    results = []
    for gene in sorted(common):
        a = df_gbm[df_gbm['Hugo_Symbol']==gene]['sample'].nunique()
        b = df_COAD[df_COAD['Hugo_Symbol']==gene]['sample'].nunique()
        t1, t2 = df_gbm['sample'].nunique(), df_COAD['sample'].nunique()
        orv, pv = fisher_exact([[a,b],[t1-a,t2-b]])
        results.append({'Gene':gene,'OR':round(orv,2),'p-value':pv})
    res_df = pd.DataFrame(results).sort_values('p-value')
    res_df.to_csv(os.path.join(tables_dir,"gene_enrichment_results.csv"), index=False)
    res_df.to_excel(os.path.join(tables_dir,"gene_enrichment_results.xlsx"), index=False)

    comb = []
    for name, d in [('GBM',df_gbm),('COAD',df_COAD)]:
        tmp = (d[d['Hugo_Symbol'].isin(common)]
               .groupby('Hugo_Symbol')['sample'].nunique()
               .reset_index(name='Sample_Count')
               .assign(Cancer_Type=name)
               .rename(columns={'Hugo_Symbol':'Gene'}))
        comb.append(tmp)
    comb_df = pd.concat(comb,ignore_index=True)
    plt.figure(figsize=(8,4))
    sns.barplot(x='Gene',y='Sample_Count',hue='Cancer_Type',
                data=comb_df,palette=['#87CEEB','#D3D3D3'],width=0.6)
    plt.title("Common Top Mutated Genes in GBM and COAD",fontsize=12)
    plt.xticks(rotation=45,ha='right',fontsize=9)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(plots_dir,f"common_genes_comparison.{ext}"),dpi=900)
    plt.close()
else:
    print("❗ No common top 10 genes")

# ── 9) Signature heatmaps ──────────────────────────────────────────────────────
def plot_signature_heatmap(df, cancer_type):
    sbs_cols = sorted([c for c in df.columns if c.startswith('SBS')])
    means = df[sbs_cols].astype(float).mean().sort_values(ascending=False).head(15)
    sig_df = pd.DataFrame({
        'Signature': means.index,
        'Mean_Contribution': means.values,
        'Etiology': [cosmic_signature_map.get(sig,'Unknown') for sig in means.index]
    })
    sig_df.to_csv(os.path.join(tables_dir,f"{cancer_type}_signature_contributions.csv"), index=False)
    sig_df.to_excel(os.path.join(tables_dir,f"{cancer_type}_signature_contributions.xlsx"), index=False)
    plt.figure(figsize=(12,6))
    sns.heatmap(means.to_frame().T, annot=True, fmt=".2f",
                cbar_kws={'label':'Mean Contribution'})
    plt.title(f"Top 15 COSMIC Signatures in {cancer_type}",fontsize=12)
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(plots_dir,f"{cancer_type}_signature_heatmap.{ext}"),dpi=900)
    plt.close()

plot_signature_heatmap(df_gbm, 'GBM')
plot_signature_heatmap(df_COAD,'COAD')

# ── 10) Co-mutation patterns ────────────────────────────────────────────────────
def analyze_co_mutation(df, cancer_type, top_n=5):
    top = get_top_genes(df, top_n)['Gene'].tolist()
    samp = df['sample'].unique()
    mat = pd.DataFrame(index=samp)
    for g in top:
        mat[g] = mat.index.isin(df[df['Hugo_Symbol']==g]['sample']).astype(int)

    co = pd.DataFrame(index=top,columns=top,dtype=float)
    for g1 in top:
        for g2 in top:
            if g1==g2:
                co.loc[g1,g2]=1.0
            else:
                both = ((mat[g1]&mat[g2])==1).sum()
                either=((mat[g1]|mat[g2])==1).sum()
                co.loc[g1,g2] = both/either if either>0 else 0
    plt.figure(figsize=(8,6))
    sns.heatmap(co,annot=True,fmt=".2f",vmin=0,vmax=1)
    plt.title(f"Co-mutation in {cancer_type}",fontsize=12)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(plots_dir,f"{cancer_type}_comutation_heatmap.{ext}"),dpi=900)
    plt.close()
    return co

gbm_co  = analyze_co_mutation(df_gbm,'GBM')
coad_co = analyze_co_mutation(df_COAD,'COAD')

# ── 11) Mutational burden ───────────────────────────────────────────────────────
def compare_mutational_burden(df1,df2,n1,n2):
    b1 = df1.groupby('sample').size().reset_index(name='mutation_count')
    b2 = df2.groupby('sample').size().reset_index(name='mutation_count')
    b1['Cancer_Type'],b2['Cancer_Type'] = n1,n2
    comb = pd.concat([b1,b2],ignore_index=True)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Cancer_Type',y='mutation_count',data=comb)
    plt.title("Mutational Burden Comparison",fontsize=12)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(plots_dir,f"mutational_burden_comparison.{ext}"),dpi=900)
    plt.close()

    stat,p = mannwhitneyu(b1['mutation_count'],b2['mutation_count'])
    print(f"Mann-Whitney U {n1} vs {n2}: U={stat:.1f}, p={p:.4f}")

    stats = pd.DataFrame({
        'Cancer_Type':[n1,n2],
        'Median_Mutations':[b1['mutation_count'].median(),b2['mutation_count'].median()],
        'Mean_Mutations':[b1['mutation_count'].mean(),b2['mutation_count'].mean()],
        'Mann_Whitney_p':[np.nan,p]
    })
    stats.to_csv(os.path.join(tables_dir,"mutational_burden_stats.csv"), index=False)
    stats.to_excel(os.path.join(tables_dir,"mutational_burden_stats.xlsx"), index=False)
    return comb

burden = compare_mutational_burden(df_gbm,df_COAD,'GBM','COAD')

# ── 12) Survival analysis (with p-value and medians on plot!) ──────────────────
def perform_survival_analysis(df, cancer_type, top_n=5, save_individual=True):
    clinical = (df[['sample','Overall Survival (Months)','Overall Survival Status']]
                .drop_duplicates('sample')
                .dropna(subset=['Overall Survival (Months)','Overall Survival Status'])
                .reset_index(drop=True))
    top_genes = get_top_genes(df,top_n)['Gene']
    ok=[]
    for gene in top_genes:
        muts = set(df[df['Hugo_Symbol']==gene]['sample'])
        surv = clinical.assign(mutated=clinical['sample'].isin(muts).astype(int))
        if surv['mutated'].sum()>=3 and (surv['mutated']==0).sum()>=3:
            ok.append(gene)
        else:
            print(f"⚠️ Skipping {gene} in {cancer_type}")

    if not ok:
        print(f"No genes passed filtering for survival in {cancer_type}")
        return

    pdf_path = os.path.join(plots_dir,f"{cancer_type}_survival_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        for gene in ok:
            surv = clinical.assign(mutated=clinical['sample'].isin(
                df[df['Hugo_Symbol']==gene]['sample']).astype(int))
            surv = surv.dropna(subset=['Overall Survival (Months)','Overall Survival Status'])

            mut_times  = surv.loc[surv['mutated']==1, 'Overall Survival (Months)']
            mut_event  = surv.loc[surv['mutated']==1, 'Overall Survival Status']
            wt_times   = surv.loc[surv['mutated']==0, 'Overall Survival (Months)']
            wt_event   = surv.loc[surv['mutated']==0, 'Overall Survival Status']

            kmf = KaplanMeierFitter()
            plt.figure(figsize=(8,6))

            # Mut group (red)
            kmf.fit(mut_times, mut_event, label=f"{gene} Mut")
            ax = kmf.plot(ci_show=True, color="red")
            median_mut = kmf.median_survival_time_

            # WT group (blue)
            kmf.fit(wt_times, wt_event, label=f"{gene} WT")
            kmf.plot(ax=ax, ci_show=True, color="blue")
            median_wt = kmf.median_survival_time_

            # Log-rank test
            res = logrank_test(mut_times, wt_times, mut_event, wt_event)
            logrank_p = res.p_value

            # Add median survival lines/text
            plt.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
            plt.text(median_wt, 0.53, f"Median (WT): {median_wt:.1f} mo", color='blue', fontsize=10, ha='center', fontweight='bold')
            plt.text(median_mut, 0.48, f"Median (Mut): {median_mut:.1f} mo", color='red', fontsize=10, ha='center', fontweight='bold')

            # Log-rank p-value annotation
            plt.text(0, 0.17, f"Log-rank p = {logrank_p:.4f}", fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

            plt.title(f"{gene} Survival in {cancer_type} (p={logrank_p:.4f})")
            plt.xlabel("Months")
            plt.ylabel("Survival Probability")
            plt.tight_layout()
            pdf.savefig()
            # --- Save each survival plot as PNG and PDF too ---
            if save_individual:
                for ext in ['png', 'pdf']:
                    outpath = os.path.join(plots_dir, f"{cancer_type}_survival_{gene}.{ext}")
                    plt.savefig(outpath, dpi=900)
            plt.close()

    print(f"✅ Survival curves saved to {pdf_path}")

perform_survival_analysis(df_gbm,'GBM')
perform_survival_analysis(df_COAD,'COAD')

# ── 13) Pathway enrichment: top 5 terms ─────────────────────────────────────────
try:
    def run_pathway_top5(df,cancer_type,top_n=10,top_terms=5,p_cutoff=1.0):
        genes = get_top_genes(df,top_n)['Gene'].tolist()
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=['KEGG_2021_Human','Reactome_2022','GO_Biological_Process_2023'],
            organism='human'
        )
        enr.results.to_csv(os.path.join(tables_dir,f"{cancer_type}_pathway_full.csv"),index=False)
        enr.results.to_excel(os.path.join(tables_dir,f"{cancer_type}_pathway_full.xlsx"),index=False)
        for db in enr.results['Gene_set'].unique():
            sub = (enr.results[enr.results['Gene_set']==db]
                   .sort_values('Adjusted P-value')
                   .head(top_terms))
            if sub.empty:
                print(f"⚠️ No terms in {db} for {cancer_type}")
                continue
            print(f"🔎 {cancer_type} → {db}: plotting top {len(sub)} terms")
            plt.figure(figsize=(8,6))
            sns.scatterplot(data=sub, x='Term', y='Adjusted P-value', size='Overlap', hue='Adjusted P-value', palette="viridis", legend=False, sizes=(20,200))
            plt.title(f"{cancer_type} — {db} (top {len(sub)})")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            for ext in ['png', 'pdf']:
                out_fig = os.path.join(plots_dir, f"{cancer_type}_{db}_top{len(sub)}.{ext}")
                plt.savefig(out_fig, dpi=900)
            plt.close()
    run_pathway_top5(df_gbm,'GBM', top_n=10, top_terms=5)
    run_pathway_top5(df_COAD,'COAD', top_n=10, top_terms=5)

except ImportError:
    print("⚠️ gseapy missing — skipping pathway")
except Exception as e:
    print("⚠️ Pathway analysis failed:", e)

print("\n✅ All analysis complete — check 'plots/' and 'tables/'")
