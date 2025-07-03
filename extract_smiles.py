#!/usr/bin/env python3

import pandas as pd

# Read the TSV file
df = pd.read_csv('output/lm_2000_t5_linker/run_population.tsv', sep='\t')

# Extract the smiles column
smiles_df = pd.DataFrame({'smiles': df['smiles']})

# Save to a new file in the desired format
smiles_df.to_csv('smiles.csv', index=False)

print(f"Extracted {len(smiles_df)} SMILES strings to smiles.csv") 