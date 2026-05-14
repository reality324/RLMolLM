#!/usr/bin/env python3
"""Optimize molecules using two different checkpoint files - Non-scaffold Version"""

import sys
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

import os
import json
import pandas as pd
import torch
from datetime import datetime

# Read initial molecules
initial_population_file = "/home/tianwangcong/RLMolLM/assets/initial_populations/canonical_two_mols.csv"
initial_df = pd.read_csv(initial_population_file)
print(f"Initial molecules: {len(initial_df)}")
for i, smiles in enumerate(initial_df['smiles']):
    print(f"  {i+1}. {smiles[:60]}...")

# Checkpoint files to test
checkpoints = [
    ("/home/tianwangcong/RLMolLM/assets/models/moses_pretrained_fixed.pt", "checkpoint_fixed"),
    ("/home/tianwangcong/RLMolLM/assets/models/moses_pretrained.pt", "checkpoint_original"),
]

results = []

for checkpoint_path, checkpoint_name in checkpoints:
    print("\n" + "=" * 70)
    print(f"Testing with: {checkpoint_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    output_dir = f"/home/tianwangcong/RLMolLM/optimization_output/{checkpoint_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from rlmollm import RLMolLMGenerator
        
        # Initialize generator
        print("Initializing generator...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        generator = RLMolLMGenerator(
            checkpoint_path=checkpoint_path,
            config_path="/home/tianwangcong/RLMolLM/assets/configs/moses.json",
            device=device,
            verbose=True,
            seed=42
        )
        print("Generator initialized successfully!")
        
        # Use non-scaffold optimization with initial population
        print("\nRunning non-scaffold optimization...")
        
        opt_results = generator.optimize(
            target_properties={'qed': 1.0, 'logp': 2.5},
            model_type='lm',
            use_scaffold=False,
            initial_population_file=initial_population_file,
            population_size=50,
            generations=3,
            output_dir=output_dir,
            return_dataframe=True,
            auto_convert_chiral=True
        )
        
        print(f"\nGenerated {len(opt_results)} molecules")
        if len(opt_results) > 0:
            best = opt_results.nlargest(5, 'fitness')
            print(f"\nTop 5 molecules by fitness:")
            for idx, row in best.iterrows():
                print(f"  Fitness: {row['fitness']:.4f} | SMILES: {row['smiles'][:60]}...")
        
        # Save results
        results_file = f"{output_dir}/final_results.csv"
        opt_results.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        results.append({
            "checkpoint": checkpoint_name,
            "status": "SUCCESS",
            "total_molecules": len(opt_results),
            "best_fitness": opt_results['fitness'].max() if len(opt_results) > 0 else 0,
            "avg_fitness": opt_results['fitness'].mean() if len(opt_results) > 0 else 0,
            "results_file": results_file
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "checkpoint": checkpoint_name,
            "status": "FAILED",
            "error": str(e)
        })

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for r in results:
    print(f"\n{r['checkpoint']}:")
    print(f"  Status: {r['status']}")
    if r['status'] == 'SUCCESS':
        print(f"  Total molecules: {r['total_molecules']}")
        print(f"  Best fitness: {r['best_fitness']:.4f}")
        print(f"  Avg fitness: {r['avg_fitness']:.4f}")
        print(f"  Results file: {r['results_file']}")
    else:
        print(f"  Error: {r.get('error', 'Unknown')}")
