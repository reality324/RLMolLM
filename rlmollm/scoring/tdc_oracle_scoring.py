"""
TDC Oracle Scoring for RLMolLM

This module provides TDC (Therapeutics Data Commons) oracle support
for molecular optimization, matching InVirtuoGen's oracle system.
"""

import sys
import numpy as np
from typing import List, Dict, Optional

# Add RDKit compatibility for TDC
class _RDKitSixCompat:
    """RDKit six module compatibility layer for RDKit 2024+"""
    @staticmethod
    def iteritems(d):
        return iter(d.items())
    @staticmethod
    def string_types():
        return (str,)

# Inject compatibility before importing TDC
sys.modules['rdkit.six'] = _RDKitSixCompat()

import tdc
from rdkit import Chem
from rdkit.Chem import QED, Crippen

from rlmollm.scoring.scoring_interface import ScoringInterface


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

# Target molecules for similarity/MPO oracles
ORACLE_TARGETS = {
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


class TDCOracleScoring(ScoringInterface):
    """Scoring interface using TDC Oracles for molecular evaluation.
    
    This matches the oracle system used in InVirtuoGen for fair comparison.
    """
    
    def __init__(
        self, 
        oracle_name: str,
        scoring_parameters: Optional[Dict] = None,
        data_column_name: str = 'smiles',
        fitness_column_name: str = 'fitness'
    ):
        """Initialize TDC Oracle Scoring.
        
        Args:
            oracle_name: Name of the TDC oracle (e.g., 'qed', 'drd2', 'jnk3')
            scoring_parameters: Additional parameters (e.g., target molecule for similarity)
            data_column_name: Name of the data column
            fitness_column_name: Name of the fitness column
        """
        super().__init__()
        
        if oracle_name not in TDC_ORACLES:
            raise ValueError(
                f"Unknown oracle: {oracle_name}. "
                f"Available oracles: {TDC_ORACLES}"
            )
        
        self._oracle_name = oracle_name
        self._data_column_name = data_column_name
        self._fitness_column_name = fitness_column_name
        
        # Initialize TDC oracle
        try:
            self._tdc_oracle = tdc.Oracle(oracle_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TDC oracle '{oracle_name}': {e}")
        
        # Get target molecule if needed
        self._target_mol = None
        if oracle_name in ORACLE_TARGETS:
            self._target_mol = Chem.MolFromSmiles(ORACLE_TARGETS[oracle_name])
        
        # Property name for this oracle
        self._scoring_name = f"tdc_{oracle_name}"
        
    @property
    def column_names(self) -> List[str]:
        return [self._scoring_name]
    
    @property
    def selection_names(self) -> List[str]:
        return [self._scoring_name]
    
    @property
    def data_column_name(self) -> str:
        return self._data_column_name
    
    @property
    def fitness_column_name(self) -> str:
        return self._fitness_column_name
    
    def prepare_data_for_scoring(self, sequence: str):
        """Prepare SMILES string for scoring.
        
        Args:
            sequence: SMILES string
            
        Returns:
            RDKit Mol object or None if invalid
        """
        if sequence is None:
            return None
        try:
            mol = Chem.MolFromSmiles(sequence)
            return mol
        except:
            return None
    
    def make_canonical(self, sequence) -> str:
        """Convert prepared sequence to canonical SMILES.
        
        Args:
            sequence: RDKit Mol object
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        if sequence is None:
            return None
        try:
            return Chem.MolToSmiles(sequence, canonical=True)
        except:
            return None
    
    def generate_scores(self, mols: List) -> Dict[str, List]:
        """Generate TDC oracle scores for molecules.

        Args:
            mols: List of RDKit Mol objects

        Returns:
            Dictionary with smiles and oracle scores
        """
        scores = []
        smiles_list = []

        for mol in mols:
            if mol is None:
                scores.append(0.0)
                smiles_list.append("")
                continue

            try:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
                score = self._tdc_oracle(smiles)
                if score is None or np.isnan(score):
                    scores.append(0.0)
                else:
                    scores.append(float(score))
            except Exception as e:
                smiles_list.append("")
                scores.append(0.0)

        return {
            self._data_column_name: smiles_list,
            self._scoring_name: scores
        }
    
    def get_oracle_name(self) -> str:
        """Get the TDC oracle name."""
        return self._oracle_name


def create_tdc_oracle_scoring(
    oracle_name: str,
    **kwargs
) -> TDCOracleScoring:
    """Factory function to create TDC Oracle Scoring.
    
    Args:
        oracle_name: Name of the TDC oracle
        **kwargs: Additional arguments for TDCOracleScoring
        
    Returns:
        TDCOracleScoring instance
    """
    return TDCOracleScoring(oracle_name=oracle_name, **kwargs)


def get_all_tdc_oracles() -> List[str]:
    """Get list of all available TDC oracles."""
    return TDC_ORACLES.copy()


def evaluate_molecules_with_oracles(
    molecules: List[str], 
    oracle_names: List[str]
) -> Dict[str, Dict]:
    """Evaluate molecules with multiple TDC oracles.
    
    Args:
        molecules: List of SMILES strings
        oracle_names: List of TDC oracle names
        
    Returns:
        Dictionary mapping oracle names to score dictionaries
    """
    results = {}
    
    for oracle_name in oracle_names:
        if oracle_name not in TDC_ORACLES:
            print(f"Warning: Unknown oracle '{oracle_name}', skipping")
            continue
        
        try:
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
