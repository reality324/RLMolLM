#!/usr/bin/env python3
"""Test hERG property optimization with a custom initial molecule."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add workspace to path
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path, get_config_path

def main():
    smiles = "O=C(C)N1CCN(C2C=C3C(C4=C(C(N[C@@H]5[C@@H](C(=O)N)C6CCC5CC6)=CN=C4)CO3)=CC=2)CC1"

    # Create initial population file
    initial_pop_df = pd.DataFrame({'smiles': [smiles]})
    init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
    initial_pop_df.to_csv(init_file, index=False)

    print("=" * 60)
    print("RLMolLM hERG Property Optimization Test")
    print("=" * 60)
    print(f"\nInitial molecule: {smiles}")

    # Initialize generator
    print("\n[1] Initializing RLMolLMGenerator...")
    generator = RLMolLMGenerator(
        checkpoint_path=get_model_path("moses"),
        config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
        device="cuda"
    )
    print("    Generator initialized successfully!")

    # Define target properties to optimize
    # hERG_Blockers: preferred_value = 0.0 (want low hERG blocking = safer)
    # Note: 'hERG' maps to 'hERG_Blockers' internally
    target_properties = {
        'hERG': 0.0,   # Minimize hERG blocking
    }
    print(f"\n[2] Target properties: {target_properties}")

    # Run optimization
    print("\n[3] Starting optimization...")
    print("    - Population size: 200")
    print("    - Generations: 10")
    print("-" * 60)

    molecules_df = generator.optimize(
        target_properties=target_properties,
        initial_population_file=init_file,
        population_size=100,
        generations=3,
        use_scaffold=False,
        mutation_parameter=0.4,
        output_dir="/home/tianwangcong/RLMolLM/optimization_results",
        return_dataframe=True,
        batch_size=32,
        top_k=5,
        lr=0.00002,
        auto_convert_chiral=False
    )

    print("-" * 60)
    print("\n[4] Optimization complete!")
    print(f"    Generated {len(molecules_df)} molecules")

    # Show top results
    if molecules_df is not None and len(molecules_df) > 0:
        print("\n[5] Top 10 molecules by hERG score (lower is better):")
        print("-" * 60)

        top_by_herg = molecules_df.nsmallest(10, 'hERG_Blockers_raw')
        for i, row in enumerate(top_by_herg.itertuples(), 1):
            herg_raw = getattr(row, 'hERG_Blockers_raw', 'N/A')
            herg_norm = getattr(row, 'hERG_Blockers', 'N/A')
            smi = getattr(row, 'smiles', 'N/A')
            print(f"  {i:2d}. hERG_raw={herg_raw:.4f} hERG_norm={herg_norm:.4f}  SMILES: {smi[:60]}...")

        # Save results
        output_file = "/home/tianwangcong/RLMolLM/optimization_results/molecules.csv"
        molecules_df.to_csv(output_file, index=False)
        print(f"\n[6] Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Optimization finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
