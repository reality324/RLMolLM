#!/usr/bin/env python3
"""Optimize molecules using ADMET properties with better initialization"""

import sys
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import os
import json
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Target molecule and similar molecules for initial population
target_smiles = "CC(C)[C@H]1CC[C@H](C)C[C@@H]1O"
print(f"Target molecule: {target_smiles}")

# Create initial population with variations
initial_molecules = [
    target_smiles,
    "CC(C)CC(O)C1CCCC1",           # Similar structure
    "CC(C)[C@H]1CC[C@@H](C)C[C@@H]1O",  # Slightly different
    "CC(C)[C@@H]1CC[C@H](C)C[C@@H]1O",  # Different stereochemistry
    "CC(C)CC[C@@H](C)C[C@@H]1O",   # Open ring
    "O[C@H]1CC[C@@H](C(C)C)C1",     # Simplified
    "CC(C)[C@H]1CCC[C@@H]1O",       # Smaller
    "CC(C)[C@H]1CC[C@H](C)C1",      # Without OH
    "CC(C)[C@H]1CC[C@H](CO)C1",     # With extra O
    "CC(C)[C@H]1CC[C@H](C(=O)O)C1", # With acid
]

# Filter valid molecules
valid_mols = []
for smiles in initial_molecules:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        valid_mols.append(smiles)
        print(f"  Valid: {smiles}")
    else:
        print(f"  Invalid: {smiles}")

print(f"\n{len(valid_mols)} valid molecules for initial population")

# Save initial population
initial_population_file = "/home/tianwangcong/RLMolLM/assets/initial_populations/target_mol.csv"
os.makedirs(os.path.dirname(initial_population_file), exist_ok=True)
with open(initial_population_file, 'w') as f:
    f.write("smiles\n")
    for smiles in valid_mols:
        f.write(smiles + "\n")

# Target properties
target_properties = {
    'QED': 1.0,                      # Drug-likeness
    'synth': 1.0,                    # Synthetic accessibility
    'BBB': 1.0,                      # Blood-brain barrier
    'Solubility_AqSolDB': -1.0,      # Preferred value -1
    'CYP3A4_inhibitor': 0.0,         # Minimize
    'AMES_Mutagenicity': 0.0,        # Minimize
    'DILI': 0.0,                     # Minimize
    'hERG': 0.0,                     # Minimize
}

checkpoint_path = "/home/tianwangcong/RLMolLM/assets/models/moses_pretrained.pt"
output_dir = "/home/tianwangcong/RLMolLM/optimization_output/liten_admet_test"
os.makedirs(output_dir, exist_ok=True)

try:
    from rlmollm import RLMolLMGenerator
    
    print("\nInitializing generator...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    generator = RLMolLMGenerator(
        checkpoint_path=checkpoint_path,
        config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
        device=device,
        verbose=True,
        seed=42
    )
    print("Generator initialized!")
    
    print("\nRunning optimization...")
    print(f"Target properties: {target_properties}")
    
    opt_results = generator.optimize(
        target_properties=target_properties,
        model_type='lm',
        use_scaffold=False,
        initial_population_file=initial_population_file,
        population_size=200,
        generations=10,
        output_dir=output_dir,
        return_dataframe=True,
        auto_convert_chiral=True
    )
    
    print(f"\n{'='*70}")
    print(f"Generated {len(opt_results)} molecules")
    
    if len(opt_results) > 0:
        print(f"\nTop 10 molecules by fitness:")
        best = opt_results.nlargest(10, 'fitness')
        for rank, (idx, row) in enumerate(best.iterrows(), 1):
            print(f"\n  === Rank {rank} ===")
            print(f"  Fitness: {row['fitness']:.4f}")
            smiles = row['smiles']
            print(f"  SMILES: {smiles}")
            # Show properties
            props_to_show = ['QED', 'BBB', 'synth', 'DILI', 'hERG', 'AMES_Mutagenicity']
            for prop in props_to_show:
                if prop in row and pd.notna(row.get(prop)):
                    print(f"    {prop}: {row[prop]:.4f}")
    
    # Save results
    results_file = f"{output_dir}/final_results.csv"
    opt_results.to_csv(results_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
