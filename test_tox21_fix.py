#!/usr/bin/env python3
"""Quick test for tox21 properties after fix."""
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

# Test one tox21 property
print("\nTesting NR_AhR (tox21_cla)...")
try:
    molecules_df = generator.optimize(
        target_properties={'NR_AhR': 1.0},
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
        if 'NR_AhR' in molecules_df.columns:
            val = molecules_df['NR_AhR'].iloc[0]
            if pd.notna(val):
                print(f"✅ SUCCESS: NR_AhR = {val:.4f}")
            else:
                print(f"❌ ERROR: NR_AhR is NaN")
        else:
            print(f"❌ ERROR: Missing column NR_AhR")
    else:
        print(f"❌ ERROR: No molecules generated")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
