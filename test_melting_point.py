#!/usr/bin/env python3
"""Test all regression properties with moses.json config."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path

def test_property(property_name, target_value, source, generator):
    """Test a single regression property."""
    print(f"\n  Testing {property_name} (target: {target_value}, source: {source})...")
    
    init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
    initial_pop_df = pd.DataFrame({'smiles': ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]})
    initial_pop_df.to_csv(init_file, index=False)
    
    # RDKit uses lowercase column names
    col_name = property_name.lower() if source == "rdkit" else property_name
    raw_col = col_name  # RDKit doesn't have _raw suffix
    
    try:
        molecules_df = generator.optimize(
            target_properties={property_name: target_value},
            initial_population_file=init_file,
            population_size=20,
            generations=1,
            use_scaffold=False,
            mutation_parameter=0.4,
            output_dir="/home/tianwangcong/RLMolLM/optimization_results",
            return_dataframe=True,
            batch_size=32,
            top_k=3,
            lr=0.00002,
            auto_convert_chiral=False
        )
        
        if molecules_df is not None and len(molecules_df) > 0:
            if col_name in molecules_df.columns:
                val = molecules_df[col_name].iloc[0]
                print(f"    SUCCESS: {col_name}={val:.4f}")
                return True
            else:
                print(f"    ERROR: Missing column {col_name}, available: {list(molecules_df.columns)}")
                return False
        else:
            print(f"    ERROR: No molecules generated")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def main():
    smiles = "O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"

    print("=" * 60)
    print("RLMolLM All Regression Properties Test")
    print("=" * 60)
    print(f"\nInitial molecule: {smiles}")

    print("\n[1] Initializing RLMolLMGenerator with moses.json...")
    generator = RLMolLMGenerator(
        checkpoint_path=get_model_path("moses"),
        config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
        device="cuda"
    )
    print("    Generator initialized successfully!")

    properties = [
        ("logS", -1.0, "liten"),
        ("logD7.4", 2.0, "liten"),
        ("logP", 2.5, "rdkit"),
        ("Melting_point", 150.0, "liten"),
        ("Boiling_point", 250.0, "liten"),
        ("pKa_acidic", 9.0, "liten"),
        ("pKa_basic", 8.0, "liten"),
    ]

    print("\n[2] Testing all regression properties...")
    print("-" * 60)
    
    results = {}
    for prop_name, target, source in properties:
        results[prop_name] = test_property(prop_name, target, source, generator)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for prop_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {prop_name}: {status}")
    
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
