import pandas as pd

df = pd.read_csv("mutations_enriched.tsv", sep="\t")

# Add 'chr' prefix for CHASMplus input
df['Chromosome'] = "chr" + df['Chromosome'].astype(str)

chasm_input = df[[
    'Chromosome',
    'Start_Position',
    'Reference_Allele',
    'Tumor_Seq_Allele2',
    'Hugo_Symbol',
    'Variant_Classification'
]]

chasm_input.to_csv("chasm_input.tsv", sep="\t", index=False)
print("✅ CHASMplus input file saved as chasm_input.tsv")
