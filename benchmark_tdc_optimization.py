#!/usr/bin/env python3
"""
RLMolLM TDC Oracle Optimization Benchmark

This script runs RLMolLM optimization on all 23 TDC oracles (same as InVirtuoGen)
for fair comparison.

Usage:
    python benchmark_tdc_optimization.py --oracle jnk3 --generations 10
    python benchmark_tdc_optimization.py --all_oracles --generations 5
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

# RDKit compatibility for TDC
class _RDKitSixCompat:
    @staticmethod
    def iteritems(d):
        return iter(d.items())
    @staticmethod
    def string_types():
        return (str,)

sys.modules['rdkit.six'] = _RDKitSixCompat()

# Apply sklearn compatibility patch for jnk3/gsk3b models
from rlmollm.scoring.sklearn_compat import apply_sklearn_tree_patch
apply_sklearn_tree_patch()

import tdc
from rdkit import Chem

# 23 TDC Oracles matching InVirtuoGen
TDC_ORACLES = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
]


def get_oracle_target(oracle_name: str) -> Optional[str]:
    """Get target molecule SMILES for similarity/MPO oracles."""
    targets = {
        "albuterol_similarity": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "amlodipine_mpo": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
        "celecoxib_rediscovery": "CC1=CC=C(C=C1)C2=CC(=NN1C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "fexofenadine_mpo": "CC(C)(C)OC(=O)N1CCC(CC1)C2=CC=CC=C2C(C3=CC=CC=C3)OCC(=O)O",
        "osimertinib_mpo": "CN[C@H]1CN(C[C@H]1C2=NC=CC(=C2)C#N)C3=CC(=CC=C3)NC4=CC=CC=C4",
        "perindopril_mpo": "CC(C)C[C@H](C(=O)N1CCCC1C(=O)O)NC(CC2=CC=CC=C2)C(=O)O",
        "ranolazine_mpo": "CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)NCCN(CC)CC",
        "sitagliptin_mpo": "CC1=CN(C2=CN=C(N=C2C1)N3CCN(CC3)C(=O)C4CC4)C",
        "zaleplon_mpo": "CC(=O)N1C=CN=C1C2=CC=CC=N2C",
        "thiothixene_rediscovery": "CN(C)CCCN1C2=CC=CC=C2SC3=CC=CC=C13",
        "troglitazone_rediscovery": "CC1=C(C=C(C=C1)O)C(CC2=CC(=C(C=C2)O)OCC(=O)C)C3=CC=CC=C3",
        "mestranol_similarity": "C#C[C@]1(CC[C@H]2[C@@H]3CCC4=CC(=O)CCC4=C3CC[C@H]12)OC",
        "valsartan_smarts": "CC(C)C[C@@H](C(=O)N1CCC[C@H]1C(=O)O)NC(Cc2ccccc2)C(=O)O",
    }
    return targets.get(oracle_name)


def run_optimization(
    oracle_name: str,
    output_dir: str,
    population_size: int = 100,
    generations: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """Run RLMolLM optimization with TDC oracle."""
    from rlmollm import RLMolLMGenerator, get_model_path, get_config_path
    
    print(f"\n{'='*60}")
    print(f"Running RLMolLM optimization with oracle: {oracle_name}")
    print(f"{'='*60}")
    
    # Create output directory
    oracle_dir = os.path.join(output_dir, oracle_name)
    os.makedirs(oracle_dir, exist_ok=True)
    
    # Initialize generator
    generator = RLMolLMGenerator(
        checkpoint_path=get_model_path("moses"),
        config_path=get_config_path("moses"),
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        seed=seed
    )
    
    # Create initial population from a simple seed molecule
    # For fair comparison with InVirtuoGen, we use the same seed
    seed_smiles = get_oracle_target(oracle_name) or "CCO"  # Default to ethanol if no target
    
    # Create a CSV with multiple copies of seed molecule
    seed_file = os.path.join(oracle_dir, "seed_population.csv")
    num_seeds = min(50, population_size)  # Use up to 50 seed molecules
    with open(seed_file, 'w') as f:
        f.write("smiles\n")
        for _ in range(num_seeds):
            f.write(f"{seed_smiles}\n")
    
    # Run optimization with TDC oracle
    try:
        result_df = generator.optimize(
            target_properties={oracle_name: 1.0},  # Target value for normalization
            initial_population_file=seed_file,
            population_size=population_size,
            generations=generations,
            output_dir=oracle_dir,
            return_dataframe=True,
            model_type='lm',
        )
        
        # Save result
        result_file = os.path.join(oracle_dir, "optimization_results.csv")
        result_df.to_csv(result_file, index=False)
        print(f"Saved results to: {result_file}")
        
        return result_df
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def evaluate_oracles(
    molecules: List[str],
    oracle_names: List[str]
) -> Dict[str, Dict]:
    """Evaluate molecules with multiple oracles."""
    # Apply sklearn patch first
    from rlmollm.scoring.sklearn_compat import apply_sklearn_tree_patch
    apply_sklearn_tree_patch()

    import os as _os
    from rlmollm.scoring.sklearn_compat import create_tdc_compat_oracle
    _broken_oracles = {'jnk3', 'gsk3b'}

    results = {}

    for oracle_name in oracle_names:
        try:
            # Use compat wrapper for broken oracles
            if oracle_name in _broken_oracles:
                _oracle_dir = '/home/tianwangcong/RLMolLM/oracle'
                _model_path = _os.path.join(_oracle_dir, f'{oracle_name}_current.pkl')
                if _os.path.exists(_model_path):
                    oracle = create_tdc_compat_oracle(oracle_name, _model_path)
                else:
                    oracle = tdc.Oracle(oracle_name)
            else:
                oracle = tdc.Oracle(oracle_name)
            scores = []
            
            for smiles in molecules:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        scores.append(0.0)
                    else:
                        score = oracle(smiles)
                        scores.append(float(score) if score is not None else 0.0)
                except:
                    scores.append(0.0)
            
            valid_scores = [s for s in scores if s > 0]
            
            results[oracle_name] = {
                'scores': scores,
                'mean': np.mean(scores) if scores else 0,
                'max': np.max(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'top10_mean': np.mean(sorted(scores, reverse=True)[:min(10, len(scores))]) if scores else 0,
                'valid_count': len(valid_scores),
            }
        except Exception as e:
            print(f"Error with oracle '{oracle_name}': {e}")
            results[oracle_name] = {'error': str(e)}
    
    return results


def load_optimization_results(output_dir: str, oracle_name: str) -> List[str]:
    """Load molecules from optimization results."""
    result_file = os.path.join(output_dir, oracle_name, "optimization_results.csv")
    
    if not os.path.exists(result_file):
        return []
    
    try:
        df = pd.read_csv(result_file)
        if 'smiles' in df.columns:
            return df['smiles'].dropna().tolist()
    except Exception as e:
        print(f"Error loading results: {e}")
    
    return []


def create_comparison_table(
    rlmollm_results: Dict[str, Dict],
    invirtuogen_results: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """Create comparison table for all oracles."""
    rows = []
    
    for oracle_name in TDC_ORACLES:
        row = {'Oracle': oracle_name}
        
        # RLMolLM results
        if oracle_name in rlmollm_results:
            r = rlmollm_results[oracle_name]
            row['RLMolLM_Max'] = r.get('max', 0)
            row['RLMolLM_Top10'] = r.get('top10_mean', 0)
            row['RLMolLM_Mean'] = r.get('mean', 0)
        
        # InVirtuoGen results (if available)
        if invirtuogen_results and oracle_name in invirtuogen_results:
            r = invirtuogen_results[oracle_name]
            row['InVirtuoGen_Max'] = r.get('max', 0)
            row['InVirtuoGen_Top10'] = r.get('top10_mean', 0)
            row['InVirtuoGen_Mean'] = r.get('mean', 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='RLMolLM TDC Oracle Optimization Benchmark')
    parser.add_argument('--oracle', type=str, default=None, help='Single oracle to optimize')
    parser.add_argument('--all_oracles', action='store_true', help='Run all 23 oracles')
    parser.add_argument('--output_dir', type=str, default='/home/tianwangcong/RLMolLM/output/tdc_benchmark',
                       help='Output directory')
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate existing results')
    parser.add_argument('--invirtuogen_dir', type=str, default=None,
                       help='InVirtuoGen results directory for comparison')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which oracles to run
    oracles_to_run = []
    if args.oracle:
        if args.oracle not in TDC_ORACLES:
            print(f"Error: Unknown oracle '{args.oracle}'")
            print(f"Available oracles: {TDC_ORACLES}")
            return
        oracles_to_run = [args.oracle]
    elif args.all_oracles:
        oracles_to_run = TDC_ORACLES
    else:
        print("Please specify --oracle <name> or --all_oracles")
        print(f"Available oracles: {TDC_ORACLES}")
        return
    
    print(f"Will run {len(oracles_to_run)} oracle(s): {oracles_to_run}")
    
    # Run optimization for each oracle
    all_results = {}
    for oracle_name in oracles_to_run:
        if args.evaluate_only:
            # Just load existing results
            molecules = load_optimization_results(args.output_dir, oracle_name)
        else:
            # Run optimization
            result_df = run_optimization(
                oracle_name=oracle_name,
                output_dir=args.output_dir,
                population_size=args.population_size,
                generations=args.generations,
                seed=args.seed
            )
            molecules = result_df['smiles'].tolist() if 'smiles' in result_df.columns else []
        
        if molecules:
            print(f"Evaluating {len(molecules)} molecules with all 23 oracles...")
            evaluation = evaluate_oracles(molecules, TDC_ORACLES)
            all_results[oracle_name] = evaluation[oracle_name]  # Save the target oracle's result
            print(f"  {oracle_name}: max={evaluation[oracle_name].get('max', 0):.3f}, "
                  f"top10={evaluation[oracle_name].get('top10_mean', 0):.3f}")
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE (All 23 Oracles)")
    print("="*60)
    
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(args.output_dir, f"comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nSaved comparison table to: {comparison_file}")
    
    # Calculate summary statistics
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        max_scores = [r.get('max', 0) for r in all_results.values()]
        top10_scores = [r.get('top10_mean', 0) for r in all_results.values()]
        print(f"Mean Max Score: {np.mean(max_scores):.3f}")
        print(f"Mean Top-10 Score: {np.mean(top10_scores):.3f}")
        print(f"Sum of Max Scores: {np.sum(max_scores):.3f}")


if __name__ == "__main__":
    main()
