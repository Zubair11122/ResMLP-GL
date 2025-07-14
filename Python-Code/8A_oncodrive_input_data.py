import pandas as pd
df = pd.read_csv("mutations_enriched.tsv", sep="\t")
oncodrive_input = df[[
    'Chromosome',
    'Start_Position',
    'Reference_Allele',
    'Tumor_Seq_Allele2',
    'Hugo_Symbol',
    'CADD_PHRED'
]].copy()

# Remove 'chr' if exists
oncodrive_input['CHR'] = oncodrive_input['Chromosome'].astype(str).str.replace("chr", "")
oncodrive_input.rename(columns={
    'Start_Position': 'POS',
    'Reference_Allele': 'REF',
    'Tumor_Seq_Allele2': 'ALT',
    'Hugo_Symbol': 'GENE',
    'CADD_PHRED': 'SCORE'
}, inplace=True)

oncodrive_input = oncodrive_input[['CHR', 'POS', 'REF', 'ALT', 'GENE', 'SCORE']]
oncodrive_input.to_csv("oncodrive_input.tsv", sep="\t", index=False)
print("✅ OncodriveFML input file saved as oncodrive_input.tsv")
