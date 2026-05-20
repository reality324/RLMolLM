#!/usr/bin/env python3
"""Quick test for pKa properties after fix."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path

init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
initial_pop_df = pd.DataFrame({'smiles': ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]})
initial_pop_df.to_csv(init_file, index=False)

print("Initializing generator...")
generator = RLMolLMGenerator(
    checkpoint_path=get_model_path("moses"),
    config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
    device="cuda:0"
)

print("\nTesting pKa_acidic and pKa_basic...\n")

for prop_name in ['pKa_acidic', 'pKa_basic']:
    print(f"[{prop_name}]")
    
    molecules_df = generator.optimize(
        target_properties={prop_name: 2.0},
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
        raw_val = molecules_df[f'{prop_name}_raw'].iloc[0]
        norm_val = molecules_df[prop_name].iloc[0]
        print(f"  Raw: {raw_val:.4f}, Normalized: {norm_val:.4f}")
        if norm_val == 0.0:
            print(f"  ❌ STILL ZERO!")
        else:
            print(f"  ✅ OK")
    print()
