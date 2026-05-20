#!/usr/bin/env python3
"""Test all tox21_cla properties after fix."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path

tox21_properties = [
    'NR_AhR', 'NR_AR', 'NR_AR_LBD', 'NR_Aromatase',
    'NR_ER', 'NR_ER_LBD', 'NR_PPAR_gamma',
    'SR_ARE', 'SR_ATAD5', 'SR_HSE', 'SR_MMP', 'SR_p53'
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

print(f"\nTesting {len(tox21_properties)} tox21_cla properties...\n")
print("-" * 60)

results = {}
for i, prop_name in enumerate(tox21_properties, 1):
    print(f"[{i}/12] Testing {prop_name}...", end=" ")
    
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
                    print(f"✅ {val:.4f}")
                    results[prop_name] = ("PASS", val)
                else:
                    print(f"❌ NaN")
                    results[prop_name] = ("FAIL", None)
            else:
                print(f"❌ Missing column")
                results[prop_name] = ("FAIL", None)
        else:
            print(f"❌ No molecules")
            results[prop_name] = ("FAIL", None)
    except Exception as e:
        print(f"❌ {e}")
        results[prop_name] = ("FAIL", str(e))

print("-" * 60)
passed = sum(1 for v in results.values() if v[0] == "PASS")
print(f"\nResult: {passed}/{len(tox21_properties)} PASSED")

if passed < len(tox21_properties):
    failed = [k for k, v in results.items() if v[0] == "FAIL"]
    print(f"Failed: {failed}")
