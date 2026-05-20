#!/usr/bin/env python3
"""Debug pKa normalization."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import pandas as pd
from rlmollm import RLMolLMGenerator, get_model_path
from rlmollm.scoring.property_configs import PROPERTY_CONFIG

init_file = "/home/tianwangcong/RLMolLM/initial_pop.csv"
initial_pop_df = pd.DataFrame({'smiles': ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]})
initial_pop_df.to_csv(init_file, index=False)

print("Property configs for pKa:")
print(f"  pKa_acidic: {PROPERTY_CONFIG.get('pKa_acidic', 'NOT FOUND')}")
print(f"  pKa_basic: {PROPERTY_CONFIG.get('pKa_basic', 'NOT FOUND')}")

# Test with debug
generator = RLMolLMGenerator(
    checkpoint_path=get_model_path("moses"),
    config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
    device="cuda:0"
)

print("\nTesting pKa_acidic...")
molecules_df = generator.optimize(
    target_properties={'pKa_acidic': 8.0},
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

print(f"pKa_acidic: {molecules_df['pKa_acidic'].iloc[0] if 'pKa_acidic' in molecules_df.columns else 'MISSING'}")
print(f"pKa_acidic_raw: {molecules_df['pKa_acidic_raw'].iloc[0] if 'pKa_acidic_raw' in molecules_df.columns else 'MISSING'}")
