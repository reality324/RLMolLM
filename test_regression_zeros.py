#!/usr/bin/env python3
"""Test regression properties to check if they return 0."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path

# 17 regression properties
regression_properties = [
    # pcp_reg
    'logS', 'logD7.4', 'Melting_point', 'Boiling_point', 'pKa_acidic', 'pKa_basic',
    # absorption_reg
    'Caco2_Permeability', 'MDCK_Permeability',
    # distribution_reg
    'Fu', 'PPB', 'VDss',
    # excretion_reg
    'Cl_Plasma', 'T12',
    # toxicity_reg
    'BCF', 'IGC50', 'LC50DM', 'LC50FM',
]

init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
initial_pop_df = pd.DataFrame({'smiles': ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]})
initial_pop_df.to_csv(init_file, index=False)

print("Initializing generator...")
generator = RLMolLMGenerator(
    checkpoint_path=get_model_path("moses"),
    config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
    device="cuda:0"
)

print(f"\nTesting {len(regression_properties)} regression properties...\n")
print("-" * 60)

results = {}
for i, prop_name in enumerate(regression_properties, 1):
    print(f"[{i}/17] Testing {prop_name}...", end=" ", flush=True)
    
    try:
        molecules_df = generator.optimize(
            target_properties={prop_name: 1.0},
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
            if prop_name in molecules_df.columns:
                val = molecules_df[prop_name].iloc[0]
                if pd.notna(val):
                    if val == 0.0:
                        print(f"❌ {val:.4f} (IS ZERO!)")
                        results[prop_name] = ("ZERO", val)
                    else:
                        print(f"✅ {val:.4f}")
                        results[prop_name] = ("OK", val)
                else:
                    print(f"❌ NaN")
                    results[prop_name] = ("NaN", None)
            else:
                print(f"❌ Missing column")
                results[prop_name] = ("MISSING", None)
        else:
            print(f"❌ No molecules")
            results[prop_name] = ("EMPTY", None)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results[prop_name] = ("ERROR", str(e))

print("-" * 60)

# Summary
zero_props = [k for k, v in results.items() if v[0] == "ZERO"]
ok_props = [k for k, v in results.items() if v[0] == "OK"]

print(f"\nResult: {len(ok_props)}/17 OK, {len(zero_props)} are ZERO")

if zero_props:
    print(f"\n⚠️  Properties returning 0:")
    for p in zero_props:
        print(f"   - {p}")
