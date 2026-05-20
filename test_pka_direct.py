#!/usr/bin/env python3
"""Direct test for pKa properties."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

from rlmollm.scoring.liten_admet_scoring import LiTENADMETPredictor
import pandas as pd

smiles_list = ["O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl"]

print("Initializing LiTEN predictor...")
predictor = LiTENADMETPredictor(
    base_path="/home/tianwangcong/LiTEN-ADMET",
    ckpt_dir="/home/tianwangcong/LiTEN-ADMET/ckpt/best_models_all",
    conf_json="/home/tianwangcong/LiTEN-ADMET/config/config.json",
    num_confs=3,
    top_k_confs=1,
    batch_size=32,
    cache_size=20000,
    device="cuda:0"
)

print("\nPredicting pKa_acidic...")
try:
    df = predictor.predict(smiles_list, task_class="pcp_reg", columns=["pKa_acidic"])
    print(f"Result columns: {list(df.columns)}")
    print(f"pKa_acidic raw value: {df['pKa_acidic'].iloc[0] if 'pKa_acidic' in df.columns else 'NOT IN RESULT'}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nPredicting pKa_basic...")
try:
    df = predictor.predict(smiles_list, task_class="pcp_reg", columns=["pKa_basic"])
    print(f"Result columns: {list(df.columns)}")
    print(f"pKa_basic raw value: {df['pKa_basic'].iloc[0] if 'pKa_basic' in df.columns else 'NOT IN RESULT'}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
