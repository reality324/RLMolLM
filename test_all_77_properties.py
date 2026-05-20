#!/usr/bin/env python3
"""Test ALL 77 LiTEN properties with moses.json config."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
import json
from rlmollm import RLMolLMGenerator, get_model_path

def test_property(property_name, target_value, generator):
    """Test a single property."""
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
                    return True, val
                else:
                    print(f"    ERROR: {property_name} is NaN")
                    return False, None
            else:
                print(f"    ERROR: Missing column {property_name}")
                return False, None
        else:
            print(f"    ERROR: No molecules generated")
            return False, None
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    smiles = "O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"

    print("=" * 70)
    print("RLMolLM ALL 77 LiTEN Properties Test")
    print("=" * 70)
    print(f"\nInitial molecule: {smiles}")

    # Load config to get all LiTEN properties
    with open('/home/tianwangcong/RLMolLM/assets/configs/moses.json', 'r') as f:
        config = json.load(f)
    
    liten_names = config['scoring_operator']['scoring_liten_names']
    property_config = config['scoring_operator']['property_config']
    column_to_task = config['scoring_operator']['liten']['column_to_task_class']

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

    # All 77 properties with their target values from config
    properties = []
    for prop_name in liten_names:
        if prop_name in property_config:
            pref = property_config[prop_name].get('preferred_value', 0.0)
            properties.append((prop_name, pref, column_to_task.get(prop_name, 'unknown')))
        else:
            properties.append((prop_name, 0.0, column_to_task.get(prop_name, 'unknown')))

    print(f"\n[2] Testing ALL {len(properties)} LiTEN properties...")
    print("-" * 70)
    
    results = {}
    for i, (prop_name, target, task_class) in enumerate(properties):
        print(f"\n[{i+1}/{len(properties)}] {task_class}")
        success, val = test_property(prop_name, target, generator)
        results[prop_name] = {"success": success, "value": val, "task_class": task_class}
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY - ALL 77 LiTEN Properties")
    print("=" * 70)
    
    # Group by task class
    from collections import defaultdict
    by_class = defaultdict(list)
    for prop_name, res in results.items():
        by_class[res['task_class']].append((prop_name, res['success']))
    
    for task_class in sorted(by_class.keys()):
        props = by_class[task_class]
        passed = sum(1 for _, s in props if s)
        print(f"\n{task_class} ({passed}/{len(props)} passed):")
        for prop_name, success in sorted(props, key=lambda x: x[0]):
            status = "✓" if success else "✗"
            val = results[prop_name]['value']
            val_str = f"={val:.4f}" if val is not None else ""
            print(f"  {status} {prop_name}{val_str}")
    
    total_passed = sum(1 for v in results.values() if v['success'])
    total_failed = len(results) - total_passed
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_passed}/{len(results)} PASSED")
    if total_failed > 0:
        print(f"FAILED: {[k for k, v in results.items() if not v['success']]}")
    print(f"{'='*70}")
    
    # Save results
    with open('/home/tianwangcong/RLMolLM/test_all_77_properties_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: test_all_77_properties_results.json")

if __name__ == "__main__":
    main()
