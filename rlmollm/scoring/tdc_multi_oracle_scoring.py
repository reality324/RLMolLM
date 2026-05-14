"""
TDC Oracle Scoring for RLMolLM

This module provides comprehensive TDC (Therapeutics Data Commons) oracle support
for molecular optimization with all 23 standard oracles.
"""

import sys
import numpy as np
from typing import List, Dict, Optional, Callable

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
from rdkit.Chem import QED, Crippen, AllChem, rdMolDescriptors

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

# Oracle categories
ORACLE_CATEGORIES = {
    'similarity': ['albuterol_similarity', 'mestranol_similarity'],
    'mpo': ['amlodipine_mpo', 'fexofenadine_mpo', 'osimertinib_mpo', 'perindopril_mpo', 'ranolazine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo'],
    'rediscovery': ['celecoxib_rediscovery', 'thiothixene_rediscovery', 'troglitazone_rediscovery'],
    'smarts': ['valsartan_smarts'],
    'activity': ['drd2', 'gsk3b', 'jnk3'],
    'isomer': ['isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl'],
    'scaffold_hop': ['scaffold_hop', 'deco_hop'],
    'median': ['median1', 'median2'],
    'druglikeness': ['qed'],
}

# Target molecules for similarity/rediscovery/MPO oracles
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


class TDCMultiOracleScoring(ScoringInterface):
    """Scoring interface using TDC Oracles for molecular evaluation.
    
    This class supports all 23 TDC oracles for fair comparison with InVirtuoGen.
    It can use a single oracle as fitness or combine multiple properties.
    """
    
    def __init__(
        self, 
        scoring_tdc_names: List[str],
        scoring_names: Optional[List[str]] = None,
        scoring_admet_names: Optional[List[str]] = None,
        selection_names: Optional[List[str]] = None,
        scoring_parameters: Optional[Dict] = None,
        data_column_name: str = 'smiles',
        fitness_column_name: str = 'fitness',
        fitness_function: Optional[Callable] = None,
    ):
        """Initialize TDC Oracle Scoring.
        
        Args:
            scoring_tdc_names: List of TDC oracle names to use for scoring (e.g., ['albuterol_similarity'])
            scoring_names: RDKit property names (e.g., ['synth', 'drug'])
            scoring_admet_names: ADMET property names
            selection_names: Properties to use for fitness calculation
            scoring_parameters: Additional parameters
            data_column_name: Name of the data column
            fitness_column_name: Name of the fitness column
            fitness_function: Function to combine scores (default: mean)
        """
        super().__init__()
        
        # Validate oracle names
        for oracle_name in scoring_tdc_names:
            if oracle_name not in TDC_ORACLES:
                raise ValueError(
                    f"Unknown oracle: {oracle_name}. "
                    f"Available oracles: {TDC_ORACLES}"
                )
        
        self._scoring_tdc_names = scoring_tdc_names
        self._scoring_names = scoring_names or []
        self._scoring_admet_names = scoring_admet_names or []
        self._data_column_name = data_column_name
        self._fitness_column_name = fitness_column_name
        
        # Default fitness function: use first TDC oracle score directly
        if fitness_function is None:
            fitness_function = lambda x: x[0] if len(x) > 0 else 0.0
        self._fitness_function = fitness_function
        
        # Initialize TDC oracles (lazy loading)
        self._tdc_oracles = {}
        self._target_mols = {}
        for oracle_name in scoring_tdc_names:
            try:
                self._tdc_oracles[oracle_name] = tdc.Oracle(oracle_name)
            except Exception as e:
                print(f"Warning: Failed to initialize TDC oracle '{oracle_name}': {e}")
                self._tdc_oracles[oracle_name] = None
            
            # Get target molecule if needed
            if oracle_name in ORACLE_TARGETS:
                self._target_mols[oracle_name] = Chem.MolFromSmiles(ORACLE_TARGETS[oracle_name])
        
        # Selection names - use TDC oracles as selection criteria
        if selection_names is None:
            selection_names = scoring_tdc_names if scoring_tdc_names else []
        self._selection_names = selection_names
        
        # Cache for molecule conversion
        self._mol_cache = {}
        
    @property
    def column_names(self) -> List[str]:
        """Get column names for scoring output.
        
        Returns only column names that have corresponding data.
        """
        cols = [self._data_column_name]
        cols.extend(self._scoring_tdc_names)
        cols.extend(self._scoring_names)
        cols.extend([n + '_raw' for n in self._scoring_names])
        cols.extend(self._scoring_admet_names)
        cols.append(self._fitness_column_name)
        return cols
    
    @property
    def selection_names(self) -> List[str]:
        """Get selection property names."""
        return self._selection_names
    
    @property
    def data_column_name(self) -> str:
        """Get data column name."""
        return self._data_column_name
    
    @property
    def fitness_column_name(self) -> str:
        """Get fitness column name."""
        return self._fitness_column_name
    
    def prepare_data_for_scoring(self, smiles: str):
        """Prepare SMILES string for scoring.
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit Mol object or None if invalid
        """
        if smiles is None:
            return None
        
        # Check cache first
        if smiles in self._mol_cache:
            return self._mol_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self._mol_cache[smiles] = mol
            return mol
        except:
            return None
    
    def make_canonical(self, mol) -> Optional[str]:
        """Convert prepared sequence to canonical SMILES.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def generate_scores(self, mols: List) -> Dict[str, List]:
        """Generate TDC oracle scores for molecules.

        Args:
            mols: List of RDKit Mol objects

        Returns:
            Dictionary with smiles and oracle scores
        """
        output = {self._data_column_name: [], self._fitness_column_name: []}
        
        # Add TDC oracle scores
        for oracle_name in self._scoring_tdc_names:
            output[oracle_name] = []
        
        # Add RDKit property scores
        for name in self._scoring_names:
            output[name] = []
            output[name + '_raw'] = []
        
        # Process each molecule
        fitness_scores = []
        
        for mol in mols:
            if mol is None:
                # Invalid molecule
                output[self._data_column_name].append("")
                for oracle_name in self._scoring_tdc_names:
                    output[oracle_name].append(0.0)
                for name in self._scoring_names:
                    output[name].append(0.0)
                    output[name + '_raw'].append(0.0)
                fitness_scores.append(0.0)
                continue
            
            try:
                # Get canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                output[self._data_column_name].append(canonical_smiles)
                
                # Calculate TDC oracle scores
                tdcs = []
                for oracle_name in self._scoring_tdc_names:
                    score = self._calculate_tdc_score(mol, oracle_name)
                    output[oracle_name].append(score)
                    tdcs.append(score)
                
                # Calculate RDKit property scores
                for name in self._scoring_names:
                    raw_score = self._calculate_rdkit_score(mol, name)
                    output[name + '_raw'].append(raw_score)
                    # Normalize score
                    normalized = self._normalize_rdkit_score(name, raw_score)
                    output[name].append(normalized)
                    tdcs.append(normalized)
                
                # Calculate fitness
                if tdcs:
                    fitness = self._fitness_function(tdcs)
                    fitness_scores.append(fitness)
                else:
                    fitness_scores.append(0.0)
                    
            except Exception as e:
                output[self._data_column_name].append("")
                for oracle_name in self._scoring_tdc_names:
                    output[oracle_name].append(0.0)
                for name in self._scoring_names:
                    output[name].append(0.0)
                    output[name + '_raw'].append(0.0)
                fitness_scores.append(0.0)
        
        output[self._fitness_column_name] = fitness_scores
        return output
    
    def _calculate_tdc_score(self, mol: Chem.Mol, oracle_name: str) -> float:
        """Calculate TDC oracle score for a molecule.
        
        Args:
            mol: RDKit Mol object
            oracle_name: Name of the TDC oracle
            
        Returns:
            Oracle score (0.0 if error)
        """
        if mol is None:
            return 0.0
        
        oracle = self._tdc_oracles.get(oracle_name)
        if oracle is None:
            return 0.0
        
        try:
            smiles = Chem.MolToSmiles(mol)
            score = oracle(smiles)
            if score is None or np.isnan(score):
                return 0.0
            return float(score)
        except Exception as e:
            # Handle sklearn pickle compatibility issues (e.g., jnk3 oracle)
            if 'pickle' in str(e).lower() or 'dtype' in str(e).lower():
                # Try to reinitialize the oracle for sklearn-based oracles
                try:
                    oracle = tdc.Oracle(oracle_name)
                    self._tdc_oracles[oracle_name] = oracle
                    smiles = Chem.MolToSmiles(mol)
                    score = oracle(smiles)
                    if score is None or np.isnan(score):
                        return 0.0
                    return float(score)
                except:
                    return 0.0
            return 0.0
    
    def _calculate_rdkit_score(self, mol: Chem.Mol, name: str) -> float:
        """Calculate RDKit property score.
        
        Args:
            mol: RDKit Mol object
            name: Property name ('synth', 'drug', 'logP', 'logD')
            
        Returns:
            Property score
        """
        if mol is None:
            return 0.0
        
        try:
            if name == 'synth':
                from rdkit.Contrib.SA_Score import sascorer
                return sascorer.calculateScore(mol)
            elif name == 'drug':
                return QED.qed(mol)
            elif name == 'logP':
                return Crippen.MolLogP(mol)
            elif name == 'logD':
                return Crippen.MolLogD(mol)
            elif name == 'number':
                return mol.GetNumHeavyAtoms()
            else:
                return 0.0
        except:
            return 0.0
    
    def _normalize_rdkit_score(self, name: str, raw_score: float) -> float:
        """Normalize RDKit score to 0-1 range.
        
        Args:
            name: Property name
            raw_score: Raw property score
            
        Returns:
            Normalized score
        """
        if name == 'synth':
            # SA score: lower is better, 1-10 range
            # Normalize: 10 -> 1.0, 1 -> 0.0
            return max(0.0, min(1.0, (10.0 - raw_score) / 9.0))
        elif name == 'drug':
            # QED: already 0-1
            return max(0.0, min(1.0, raw_score))
        elif name in ['logP', 'logD']:
            # LogP/LogD: target range -0.4 to 5.0
            if raw_score < -0.4:
                return max(0.0, 1.0 + (raw_score + 0.4) / 5.0)
            elif raw_score > 5.0:
                return max(0.0, 1.0 - (raw_score - 5.0) / 5.0)
            else:
                return 1.0
        elif name == 'number':
            # Heavy atoms: target 10-50
            if raw_score < 10:
                return max(0.0, raw_score / 10.0)
            elif raw_score > 50:
                return max(0.0, 1.0 - (raw_score - 50) / 50.0)
            else:
                return 1.0
        else:
            return 0.0
    
    def get_oracle_name(self) -> str:
        """Get the primary TDC oracle name."""
        return self._scoring_tdc_names[0] if self._scoring_tdc_names else ""


def create_tdc_scoring(
    oracle_name: str,
    fitness_function=None,
    **kwargs
) -> TDCMultiOracleScoring:
    """Factory function to create TDC Oracle Scoring for a single oracle.
    
    Args:
        oracle_name: Name of the TDC oracle
        fitness_function: Optional function to combine scores
        **kwargs: Additional arguments for TDCMultiOracleScoring
        
    Returns:
        TDCMultiOracleScoring instance
    """
    return TDCMultiOracleScoring(
        scoring_tdc_names=[oracle_name],
        scoring_names=kwargs.get('scoring_names', []),
        scoring_admet_names=kwargs.get('scoring_admet_names', []),
        selection_names=[oracle_name],
        fitness_column_name='fitness',
        fitness_function=fitness_function,
        **kwargs
    )


def get_all_tdc_oracles() -> List[str]:
    """Get list of all available TDC oracles."""
    return TDC_ORACLES.copy()


def get_oracle_category(oracle_name: str) -> str:
    """Get the category of an oracle."""
    for category, oracles in ORACLE_CATEGORIES.items():
        if oracle_name in oracles:
            return category
    return "unknown"


def get_oracle_info(oracle_name: str) -> Dict:
    """Get information about an oracle."""
    return {
        'name': oracle_name,
        'category': get_oracle_category(oracle_name),
        'target': ORACLE_TARGETS.get(oracle_name, None),
        'description': _ORACLE_DESCRIPTIONS.get(oracle_name, ''),
    }


# Oracle descriptions
_ORACLE_DESCRIPTIONS = {
    "albuterol_similarity": "Similarity to Albuterol (bronchodilator)",
    "amlodipine_mpo": "Multi-objective optimization for Amlodipine",
    "celecoxib_rediscovery": "Rediscovery of Celecoxib (COX-2 inhibitor)",
    "deco_hop": "Decoherence scaffold hopping",
    "drd2": "Dopamine D2 receptor activity",
    "fexofenadine_mpo": "Multi-objective optimization for Fexofenadine",
    "gsk3b": "GSK3β kinase activity",
    "isomers_c7h8n2o2": "Isomer matching for C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "Isomer matching for C9H10N2O2PF2Cl",
    "jnk3": "JNK3 kinase activity",
    "median1": "Multi-property optimization median 1",
    "median2": "Multi-property optimization median 2",
    "mestranol_similarity": "Similarity to Mestranol (estrogen)",
    "osimertinib_mpo": "Multi-objective optimization for Osimertinib",
    "perindopril_mpo": "Multi-objective optimization for Perindopril",
    "qed": "Quantitative Estimate of Drug-likeness",
    "ranolazine_mpo": "Multi-objective optimization for Ranolazine",
    "scaffold_hop": "Scaffold hopping for novelty",
    "sitagliptin_mpo": "Multi-objective optimization for Sitagliptin",
    "thiothixene_rediscovery": "Rediscovery of Thiothixene",
    "troglitazone_rediscovery": "Rediscovery of Troglitazone",
    "valsartan_smarts": "SMARTS pattern matching for Valsartan",
    "zaleplon_mpo": "Multi-objective optimization for Zaleplon",
}


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
