#!/usr/bin/env python3
"""Generate results using the best model from epoch 5"""

import sys
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import os
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

checkpoint_path = "/home/tianwangcong/RLMolLM/optimization_output/liten_admet_test/best_model_epoch_5_generator_0.pt"
output_dir = "/home/tianwangcong/RLMolLM/optimization_output/liten_admet_test"

target_properties = {
    'QED': 1.0,
    'synth': 1.0,
    'BBB': 1.0,
    'Solubility_AqSolDB': -1.0,
    'CYP3A4_inhibitor': 0.0,
    'AMES_Mutagenicity': 0.0,
    'DILI': 0.0,
    'hERG': 0.0,
}

try:
    from rlmollm import RLMolLMGenerator
    
    print("Initializing generator with best model...")
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
    
    print("\nGenerating molecules with best model...")
    print(f"Target properties: {target_properties}")
    
    # Generate molecules
    results = generator.optimize(
        target_properties=target_properties,
        model_type='lm',
        use_scaffold=False,
        population_size=500,  # Generate more molecules
        generations=1,  # Just one generation for quick results
        output_dir=output_dir,
        return_dataframe=True,
        auto_convert_chiral=True
    )
    
    print(f"\n{'='*70}")
    print(f"Generated {len(results)} molecules")
    
    if len(results) > 0:
        print(f"\nTop 10 molecules by fitness:")
        best = results.nlargest(10, 'fitness')
        for rank, (idx, row) in enumerate(best.iterrows(), 1):
            print(f"\n  === Rank {rank} ===")
            print(f"  Fitness: {row['fitness']:.4f}")
            smiles = row['smiles']
            print(f"  SMILES: {smiles}")
            props_to_show = ['QED', 'BBB', 'synth', 'DILI', 'hERG', 'AMES_Mutagenicity', 'Solubility_AqSolDB', 'CYP3A4_inhibitor']
            for prop in props_to_show:
                if prop in row and pd.notna(row.get(prop)):
                    print(f"    {prop}: {row[prop]:.4f}")
    
    results_file = f"{output_dir}/epoch5_results.csv"
    results.to_csv(results_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
