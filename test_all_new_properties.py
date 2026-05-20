#!/usr/bin/env python3
"""Test all newly added regression properties with moses.json config."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path

def test_property(property_name, target_value, generator):
    """Test a single regression property."""
    print(f"\n  Testing {property_name} (target: {target_value})...")
    
    init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
    initial_pop_df = pd.DataFrame({'smiles': ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]})
    initial_pop_df.to_csv(init_file, index=False)
    
    try:
        molecules_df = generator.optimize(
            target_properties={property_name: target_value},
            initial_population_file=init_file,
            population_size=10,
            generations=1,
            use_scaffold=False,
            mutation_parameter=0.4,
            output_dir="/home/tianwangcong/RLMolLM/optimization_results",
            return_dataframe=True,
            batch_size=16,
            top_k=3,
            lr=0.00002,
            auto_convert_chiral=False
        )
        
        if molecules_df is not None and len(molecules_df) > 0:
            if property_name in molecules_df.columns:
                val = molecules_df[property_name].iloc[0]
                if pd.notna(val):
                    print(f"    SUCCESS: {property_name}={val:.4f}")
                    return True
                else:
                    print(f"    ERROR: {property_name} is NaN")
                    return False
            else:
                print(f"    ERROR: Missing column {property_name}, available: {list(molecules_df.columns)}")
                return False
        else:
            print(f"    ERROR: No molecules generated")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    smiles = "O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"

    print("=" * 60)
    print("RLMolLM Newly Added Regression Properties Test")
    print("=" * 60)
    print(f"\nInitial molecule: {smiles}")

    print("\n[1] Initializing RLMolLMGenerator with moses.json...")
    try:
        generator = RLMolLMGenerator(
            checkpoint_path=get_model_path("moses"),
            config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
            device="cuda:0"
        )
        print("    Generator initialized successfully!")
    except Exception as e:
        print(f"    ERROR initializing generator: {e}")
        return

    # All newly added regression properties to test
    properties = [
        # pcp_reg
        ("logS", -1.0),
        ("logD7.4", 0.0),
        ("logP", 2.5),
        ("Melting_point", 150.0),
        ("Boiling_point", 200.0),
        ("pKa_acidic", 8.0),
        ("pKa_basic", 8.0),
        # absorption_reg
        ("Caco2_Permeability", 2.0),
        ("MDCK_Permeability", 2.0),
        # distribution_reg
        ("PPB", 50.0),
        ("VDss", 10.0),
        ("Fu", 50.0),
        # excretion_reg
        ("Cl_Plasma", 5.0),
        ("T12", 24.0),
        # toxicity_reg
        ("BCF", 0.0),
        ("IGC50", 8.0),
        ("LC50DM", 8.0),
        ("LC50FM", 8.0),
    ]

    print(f"\n[2] Testing {len(properties)} regression properties...")
    print("-" * 60)
    
    results = {}
    for prop_name, target in properties:
        results[prop_name] = test_property(prop_name, target, generator)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for prop_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {prop_name}: {status}")
    
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{len(properties)} tests passed")
    
    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        print(f"Failed properties: {failed}")

if __name__ == "__main__":
    main()
