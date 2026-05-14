#!/usr/bin/env python3
"""
RLMolLM 分子在 InVirtuoGen TDC Oracle 上的基准测试

测试 RLMolLM 生成的分子在 InVirtuoGen 的 Oracle 上的表现。
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# ============================================================================
# RDKit 兼容性修复 (RDKit 2024+ 移除了 rdkit.six 模块)
# TDC 使用旧版 RDKit API，需要创建兼容层
# ============================================================================
class _RDKitSixCompat:
    """RDKit six 模块兼容层，用于 RDKit 2024+ 版本"""
    @staticmethod
    def iteritems(d):
        return iter(d.items())
    @staticmethod
    def string_types():
        return (str,)

# 在导入 TDC 之前注入 rdkit.six 兼容模块
sys.modules['rdkit.six'] = _RDKitSixCompat()

# 添加路径
sys.path.insert(0, '/home/tianwangcong/InVirtuoGen_results-main')
sys.path.insert(0, '/home/tianwangcong/RLMolLM')

from rdkit import Chem
from rdkit.Chem import AllChem
import tdc


def load_rlmollm_molecules(output_dir: str, num_samples: int = 100) -> List[str]:
    """
    加载 RLMolLM 生成的分子
    
    Args:
        output_dir: RLMolLM 输出目录
        num_samples: 最多加载的分子数量
        
    Returns:
        SMILES 列表
    """
    # 优先从 final_population.csv 加载
    csv_path = os.path.join(output_dir, "final_population.csv")
    tsv_path = os.path.join(output_dir, "run_new_sequences.tsv")
    
    molecules = []
    
    # 尝试 CSV 格式
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'smiles' in df.columns:
                molecules = df['smiles'].dropna().tolist()[:num_samples]
                print(f"Loaded {len(molecules)} molecules from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    # 尝试 TSV 格式
    if not molecules and os.path.exists(tsv_path):
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            if 'smiles' in df.columns:
                molecules = df['smiles'].dropna().tolist()[:num_samples]
                print(f"Loaded {len(molecules)} molecules from {tsv_path}")
        except Exception as e:
            print(f"Error loading TSV: {e}")
    
    # 验证分子
    valid_molecules = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(Chem.MolToSmiles(mol))
    
    print(f"Valid molecules: {len(valid_molecules)} / {len(molecules)}")
    return valid_molecules


def evaluate_with_oracle(molecules: List[str], oracle_name: str) -> Dict:
    """
    使用 TDC Oracle 评估分子
    
    Args:
        molecules: SMILES 列表
        oracle_name: Oracle 名称
        
    Returns:
        评估结果字典
    """
    try:
        oracle = tdc.Oracle(oracle_name)
        
        scores = []
        for smiles in molecules:
            try:
                score = oracle(smiles)
                if score is not None and not np.isnan(score):
                    scores.append(score)
                else:
                    scores.append(0.0)
            except Exception as e:
                scores.append(0.0)
        
        return {
            'oracle': oracle_name,
            'num_molecules': len(molecules),
            'num_valid_scores': sum(1 for s in scores if s > 0),
            'scores': scores,
            'mean': np.mean(scores) if scores else 0,
            'max': np.max(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'top10_mean': np.mean(sorted(scores, reverse=True)[:10]) if len(scores) >= 10 else np.mean(scores),
        }
    except Exception as e:
        print(f"Error with oracle {oracle_name}: {e}")
        return {
            'oracle': oracle_name,
            'error': str(e),
        }


def get_all_oracles() -> List[str]:
    """获取所有可用的 Oracle 名称"""
    return [
        # MPO (Multi-Property Optimization)
        'amlodipine_mpo',
        'fexofenadine_mpo',
        'osimertinib_mpo',
        'perindopril_mpo',
        'ranolazine_mpo',
        'zaleplon_mpo',
        # Rediscovery
        'celecoxib_rediscovery',
        'troglitazone_rediscovery',
        'thiothixene_rediscovery',
        # Similarity
        'albuterol_similarity',
        'mestranol_similarity',
        # Docking
        'docking_parp1_idx0_thr4',
        # Others
        'deco_hop',
        'drd2',
        'gsk3b',
        'jnk3',
        'qed',
        'scaffold_hop',
        'valsartan_smarts',
        'isomers_c7h8n2o2',
        'isomers_c9h10n2o2pf2cl',
    ]


def run_benchmark(
    rlmollm_output_dir: str,
    oracles: Optional[List[str]] = None,
    num_samples: int = 100,
    output_dir: str = "benchmark_results"
) -> pd.DataFrame:
    """
    运行基准测试
    
    Args:
        rlmollm_output_dir: RLMolLM 输出目录
        oracles: 要测试的 Oracle 列表，None 表示全部
        num_samples: 每个 Oracle 测试的分子数量
        output_dir: 结果输出目录
        
    Returns:
        结果 DataFrame
    """
    # 加载分子
    print("=" * 60)
    print("Loading RLMolLM molecules...")
    print("=" * 60)
    molecules = load_rlmollm_molecules(rlmollm_output_dir, num_samples)
    
    if not molecules:
        print("No valid molecules found!")
        return pd.DataFrame()
    
    print(f"\nLoaded {len(molecules)} valid molecules")
    print(f"First 5 molecules:")
    for i, smi in enumerate(molecules[:5]):
        print(f"  {i+1}. {smi[:60]}...")
    
    # 获取 Oracle 列表
    if oracles is None:
        oracles = get_all_oracles()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    print("\n" + "=" * 60)
    print("Running Oracle Evaluation...")
    print("=" * 60)
    
    results = []
    for i, oracle_name in enumerate(oracles):
        print(f"\n[{i+1}/{len(oracles)}] Evaluating with {oracle_name}...")
        
        result = evaluate_with_oracle(molecules, oracle_name)
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue
        
        print(f"  Mean: {result['mean']:.4f}")
        print(f"  Max: {result['max']:.4f}")
        print(f"  Top10 Mean: {result['top10_mean']:.4f}")
        
        results.append(result)
    
    # 创建结果 DataFrame
    df = pd.DataFrame([{
        'oracle': r['oracle'],
        'num_molecules': r.get('num_molecules', 0),
        'num_valid_scores': r.get('num_valid_scores', 0),
        'mean': r.get('mean', 0),
        'max': r.get('max', 0),
        'std': r.get('std', 0),
        'top10_mean': r.get('top10_mean', 0),
    } for r in results])
    
    # 保存结果
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    
    # 保存详细分数
    for result in results:
        if 'scores' in result:
            scores_df = pd.DataFrame({
                'smiles': molecules[:len(result['scores'])],
                result['oracle']: result['scores']
            })
            scores_df.to_csv(
                os.path.join(output_dir, f"scores_{result['oracle']}.csv"),
                index=False
            )
    
    return df


def compare_with_invirtuogen(
    rlmollm_output_dir: str,
    invirtuogen_results_dir: str,
    oracle_name: str,
    num_samples: int = 100
) -> Dict:
    """
    比较 RLMolLM 和 InVirtuoGen 在同一 Oracle 上的表现
    
    Args:
        rlmollm_output_dir: RLMolLM 输出目录
        invirtuogen_results_dir: InVirtuoGen 结果目录
        oracle_name: Oracle 名称
        num_samples: 分子数量
        
    Returns:
        比较结果
    """
    # 加载 RLMolLM 分子
    molecules = load_rlmollm_molecules(rlmollm_output_dir, num_samples)
    
    # 评估 RLMolLM
    rlmollm_result = evaluate_with_oracle(molecules, oracle_name)
    
    # 查找 InVirtuoGen 结果
    plots_dir = os.path.join(invirtuogen_results_dir, "plots", "tdc", oracle_name)
    
    comparison = {
        'oracle': oracle_name,
        'rlmollm': {
            'num_molecules': len(molecules),
            'mean': rlmollm_result.get('mean', 0),
            'max': rlmollm_result.get('max', 0),
            'top10_mean': rlmollm_result.get('top10_mean', 0),
        },
        'invirtuogen': {
            'available': os.path.exists(plots_dir),
        }
    }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Benchmark RLMolLM molecules on TDC Oracles")
    
    # 输入
    parser.add_argument('--rlmollm_dir', type=str, 
                        default='/home/tianwangcong/RLMolLM/output/shixiong_fast',
                        help='RLMolLM output directory')
    parser.add_argument('--invirtuogen_dir', type=str,
                        default='/home/tianwangcong/InVirtuoGen_results-main',
                        help='InVirtuoGen results directory')
    
    # Oracle 选择
    parser.add_argument('--oracle', type=str, default=None,
                        help='Single oracle to test (default: all)')
    parser.add_argument('--oracles', type=str, nargs='+', default=None,
                        help='List of oracles to test')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of molecules to test')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with InVirtuoGen results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RLMolLM Benchmark on InVirtuoGen TDC Oracles")
    print("=" * 60)
    print(f"RLMolLM dir: {args.rlmollm_dir}")
    print(f"Oracle: {args.oracle or 'all'}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 60)
    
    # 确定要测试的 Oracle
    oracles = [args.oracle] if args.oracle else args.oracles
    
    # 运行基准测试
    results_df = run_benchmark(
        rlmollm_output_dir=args.rlmollm_dir,
        oracles=oracles,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    
    if not results_df.empty:
        print(results_df.to_string(index=False))
        
        # 保存到文件
        print(f"\nResults saved to {args.output_dir}/benchmark_results.csv")
        
        # 如果有比较选项
        if args.compare:
            print("\n" + "=" * 60)
            print("Comparison with InVirtuoGen")
            print("=" * 60)
            
            for oracle in results_df['oracle'].values:
                comparison = compare_with_invirtuogen(
                    args.rlmollm_dir,
                    args.invirtuogen_dir,
                    oracle,
                    args.num_samples
                )
                
                print(f"\n{oracle}:")
                print(f"  RLMolLM - Mean: {comparison['rlmollm']['mean']:.4f}, "
                      f"Max: {comparison['rlmollm']['max']:.4f}, "
                      f"Top10: {comparison['rlmollm']['top10_mean']:.4f}")
    else:
        print("No results to display!")


if __name__ == '__main__':
    main()
