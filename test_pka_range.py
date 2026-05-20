#!/usr/bin/env python3
"""Test pKa with multiple molecules."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/tianwangcong/RLMolLM')

from rlmollm.scoring.liten_admet_scoring import LiTENADMETPredictor
import pandas as pd

# Test with multiple molecules
smiles_list = [
    "O=C(NC1CC1)c1cc(NC(=O)C2CC=CCC2)ccc1Cl",
    "CCO",  # ethanol
    "c1ccccc1",  # benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
]

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

print("\nPredicting pKa for multiple molecules...")
df = predictor.predict(smiles_list, task_class="pcp_reg", columns=["pKa_acidic", "pKa_basic"])
print(df)
print()
print("pKa_acidic stats:", df['pKa_acidic'].describe())
print("pKa_basic stats:", df['pKa_basic'].describe())
