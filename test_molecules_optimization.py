#!/usr/bin/env python3
"""Test script to optimize specific molecules using scaffold-based generation"""

import sys
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

from rlmollm.generator import RLMolLMGenerator
from rdkit import Chem
import os

# Test molecules
molecules = [
    ("O=C(C)N1CCN(C2C=C3C(C4=C(C(N[C@@H]5[C@@H](C(=O)N)C6CCC5CC6)=CN=C4)CO3)=CC=2)CC1", "Molecule_1"),
    ("[C@@H]1(C(=O)N([H])[H])C(C2CCC1CC2)[N@@]([H])C1C=C2OCC3C(C4C=CC=C5C=4C=CC=N5)=CN=CC=3C2=CC=1", "Molecule_2")
]

print("=" * 60)
print("Molecule Optimization Test (Scaffold-Based)")
print("=" * 60)

# Initialize generator once
print("\nInitializing generator...")
generator = RLMolLMGenerator(
    checkpoint_path="/home/tianwangcong/RLMolLM/assets/models/moses_pretrained_fixed.pt",
    config_path="/home/tianwangcong/RLMolLM/assets/configs/guacamol.json",
    device="cuda",
    verbose=False
)
print("Generator initialized successfully!\n")

for smiles, name in molecules:
    print(f"\n[{name}] Original SMILES:")
    print(f"  {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ERROR: Invalid SMILES")
        continue

    canonical = Chem.MolToSmiles(mol)
    print(f"  Canonical: {canonical}")

    try:
        # Use scaffold-based optimization with the molecule as scaffold
        # Replace attachment points with # symbol
        scaffold_smiles = canonical.replace("*", "#")

        print(f"  Scaffold: {scaffold_smiles}")

        # Optimize using scaffold-based generation
        results = generator.optimize(
            target_properties={
                'qed': 1.0,       # Maximize drug-likeness
                'logp': 2.5,      # Target LogP of 2.5
                'sa': 1.0         # Target SA score of 1.0 (easier to synthesize)
            },
            model_type='lm',
            use_scaffold=True,
            scaffold_smiles=scaffold_smiles,
            population_size=50,
            generations=5,
            output_dir=f"test_output/{name}",
            return_dataframe=True,
            auto_convert_chiral=True
        )

        print(f"  Optimization completed!")
        print(f"  Generated {len(results)} molecules")

        if len(results) > 0:
            # Show best molecules
            best_scores = results.nlargest(3, 'fitness')[['smiles', 'fitness']]
            print(f"  Top 3 optimized molecules:")
            for idx, row in best_scores.iterrows():
                print(f"    - {row['smiles'][:80]}... fitness: {row['fitness']:.4f}")

    except Exception as e:
        print(f"  Error during optimization: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)
