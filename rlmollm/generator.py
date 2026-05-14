"""
High-level API for RLMolLM molecular generation and optimization.
"""
import torch
import numpy as np
import json
import os
import sys
import random
import pandas as pd
from typing import Dict, List, Union, Optional
from pathlib import Path
from rdkit import RDLogger

# Import from package
from rlmollm.population.population import Population
from rlmollm.utils.util import (
    parse_mutation_params,
    parse_train_flags,
    initialize_gan_operators,
    initialize_scaffold_handler,
    generate_initial_population_from_scaffold,
    generate_initial_population_without_scaffold,
    save_models,
)
from rlmollm.utils.training_utils import (
    train_and_evolve,
    setup_scoring_operator,
)
from rlmollm.utils.chiral_utils import has_chirality, remove_chirality


class RLMolLMGenerator:
    """
    High-level API for molecular generation using reinforcement learning-enhanced language models.
    
    This class provides a simple interface for optimizing molecules towards target properties
    using genetic algorithms with optional reinforcement learning (PPO).
    
    Example:
        >>> gen = RLMolLMGenerator(
        ...     checkpoint_path="model_weights/latest_model.pt",
        ...     config_path="config/default.json"
        ... )
        >>> molecules = gen.optimize(
        ...     target_properties={'qed': 1.0, 'logp': 2.5},
        ...     population_size=2000,
        ...     generations=20
        ... )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize the RLMolLMGenerator.
        
        Args:
            checkpoint_path: Path to model checkpoint file (.pt)
            config_path: Path to configuration JSON file
            device: Device to run on ("cuda" or "cpu")
            verbose: Whether to print progress information
            seed: Random seed for reproducibility
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.verbose = verbose
        self.seed = seed
        
        # Set random seeds for reproducibility
        self._set_random_seed(seed)
        
        # Suppress RDKit warnings if not verbose
        if not verbose:
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use default config if none provided
            self.config = self._get_default_config()

        # Fix tokenizer_directory: convert relative paths to absolute
        # The tokenizer lives at <RLMolLM>/tokenizer/ relative to the package root
        package_root = Path(__file__).parent.parent
        for operator in self.config.get("gan_operators", []):
            tok_dir = operator.get("tokenizer_directory", "")
            if not tok_dir or not os.path.isabs(tok_dir):
                operator["tokenizer_directory"] = str(package_root / "tokenizer")
                if self.verbose:
                    print(f"  Resolved tokenizer_directory: {tok_dir} -> {operator['tokenizer_directory']}")
        
        if self.verbose:
            print(f"RLMolLMGenerator initialized")
            print(f"  Device: {self.device}")
            print(f"  Checkpoint: {self.checkpoint_path}")
            if config_path:
                print(f"  Config: {self.config_path}")
    
    def _set_random_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "gan_operators": [
                {
                    "model_directory": "ibm/MoLFormer-XL-both-10pct",
                    "checkpoint_path": self.checkpoint_path,
                    "num_steps": 1
                }
            ],
            "scoring_functions": [
                {"name": "qed", "weight": 1.0},
                {"name": "logp", "weight": 1.0},
                {"name": "sa", "weight": 1.0},
                {"name": "logd", "weight": 1.0}
            ]
        }
    
    def optimize(
        self,
        target_properties: Dict[str, float],
        model_type: str = 'lm',
        initial_population_file: Optional[str] = None,
        population_size: int = 200,
        generations: int = 10,
        mutation_parameter: float = 0.4,
        use_scaffold: bool = False,
        scaffold_smiles: Optional[str] = None,
        mask_mode: str = "replace",
        scaffold_min_masks: Optional[int] = None,
        scaffold_max_masks: Optional[int] = None,
        scaffold_min_mask_per_position: int = 1,
        scaffold_max_mask_per_position: int = 3,
        output_dir: Optional[str] = None,
        return_dataframe: bool = False,
        batch_size: int = 32,
        top_k: int = 5,
        init_top_k: int = 5,
        lr: float = 0.00002,
        dataset_type: str = "guacamol",
        fitness_function=None,
        auto_convert_chiral: bool = True,
        **kwargs
    ) -> Union[List[str], pd.DataFrame]:
        """
        Optimize molecules towards target properties using genetic algorithm.
        
        Args:
            target_properties: Dictionary of target properties and their values
                              (e.g., {'qed': 1.0, 'logp': 2.5, 'sa': 1.0})
            model_type: Model configuration type
                       Options: 'lm' (default), 'lm_ppo', 'alm', 'alm_ppo', 'lm_ng', 'lm_ng_ppo'
            initial_population_file: Path to initial population CSV file
                                    Required for non-scaffold MLM-based generation
            population_size: Size of the molecular population (default: 200)
            generations: Number of GA iterations (default: 10)
            mutation_parameter: Mutation rate, range 0.4-1.0 (default: 0.4)
            use_scaffold: Whether to use scaffold-based generation
            scaffold_smiles: Scaffold SMILES string (required if use_scaffold=True)
                           Use # as attachment points (e.g., "#c1cc(#)ccc1#")
            mask_mode: Masking strategy for generation
                      Options: 'replace', 'random', 'sample_partition'
                      Note: Use 'sample_partition' to maintain scaffold constraints during evolution
            scaffold_min_masks: Minimum total [MASK] tokens across the molecule (default: auto-calculated)
                              If None, automatically set to ~1/3 of scaffold atom count
            scaffold_max_masks: Maximum total [MASK] tokens (default: auto-calculated)
                              If None, automatically set to ~scaffold atom count
            scaffold_min_mask_per_position: Minimum [MASK] tokens per attachment point (default: 1)
            scaffold_max_mask_per_position: Maximum [MASK] tokens per attachment point (default: 3)
            output_dir: Output directory for results (optional)
            return_dataframe: If True, return DataFrame with properties; if False, return list of SMILES
            batch_size: Batch size for training (default: 32)
            top_k: Number of top predictions for generator evaluation (default: 5)
            init_top_k: Number of top predictions for initial population (default: 5)
            lr: Learning rate (default: 0.00002)
            dataset_type: Dataset for empirical sampling ('moses', 'guacamol', 'zinc', 'gdb')
            fitness_function: Function to combine property scores (e.g. scipy.stats.gmean, scipy.stats.hmean).
                             If None, defaults to scipy.stats.hmean.
            auto_convert_chiral: If True and input has chirality, automatically convert to non-chiral
                               version for optimization. This avoids SMILES mutation issues with chiral
                               molecules. Set to False to use original chiral input as-is. (default: True)
            
        Returns:
            List of SMILES strings or pandas DataFrame with SMILES and properties
        """
        if self.verbose:
            print(f"\nStarting optimization...")
            print(f"  Target properties: {target_properties}")
            print(f"  Model type: {model_type}")
            print(f"  Population size: {population_size}")
            print(f"  Generations: {generations}")
            print(f"  Mutation parameter: {mutation_parameter}")
            print(f"  Fitness function: {fitness_function}")
            if use_scaffold:
                print(f"  Scaffold: {scaffold_smiles}")
        
        # Update config with target properties (pass fitness_function through)
        self._update_config_with_targets(target_properties, fitness_function=fitness_function)
        
        # Parse model type to get training flags
        generator_only, mlm_loss, use_ppo, no_merge = self._parse_model_type(model_type)
        
        # Set up mutation parameters
        mutation_samples_list = [population_size]
        mutation_parameter_list = [mutation_parameter]
        
        # Initialize GAN operators with checkpoint
        class Args:
            pass
        
        args = Args()
        args.use_ppo = use_ppo
        args.ppo_epochs = 1
        args.ppo_interval = 1
        args.clip_ratio = 0.2
        args.entropy_coef = 0.01
        args.value_coef = 0.5
        args.reward_scale = 1.5 if use_ppo else 1.0
        args.invalid_penalty = -0.9 if use_ppo else -1.0
        args.use_optimized_ppo = True
        args.random_init = False
        args.mlm_loss = mlm_loss
        args.lr = lr
        args.epochs = generations
        args.batch_size = batch_size
        args.top_k = top_k
        args.init_top_k = init_top_k
        args.population_size = population_size
        args.mutation_samples = population_size
        args.generator_only = generator_only
        args.no_merge = no_merge
        args.add_randomized_smiles = False
        args.reset_all_smiles = False
        args.use_reinforce = False
        args.use_scaffold = use_scaffold
        args.mask_mode = mask_mode
        args.dataset_type = dataset_type
        args.output_directory = output_dir
        args.run_id = 'run'
        args.data_file = initial_population_file
        args.seed = self.seed
        
        # Override saved_generator path in config with checkpoint_path
        for operator in self.config.get('gan_operators', []):
            if 'saved_generator' in operator or self.checkpoint_path:
                operator['saved_generator'] = self.checkpoint_path
        
        # Initialize GAN operators
        gan_operators = initialize_gan_operators(
            self.config, 
            self.device, 
            mutation_parameter_list, 
            args
        )
        
        # Initialize scoring operator
        scoring_operator = setup_scoring_operator(self.config, self.device)
        
        # Initialize scaffold handler if needed
        scaffold_handler = None
        if use_scaffold:
            if scaffold_smiles is None:
                raise ValueError("scaffold_smiles must be provided when use_scaffold=True")
            
            # Auto-calculate scaffold parameters if not provided
            scaffold_params = self._calculate_scaffold_params(
                scaffold_smiles,
                scaffold_min_masks,
                scaffold_max_masks,
                scaffold_min_mask_per_position,
                scaffold_max_mask_per_position
            )
            
            if self.verbose:
                print(f"  Scaffold parameters:")
                print(f"    min_masks: {scaffold_params['min_masks']}")
                print(f"    max_masks: {scaffold_params['max_masks']}")
                print(f"    min_mask_per_position: {scaffold_params['min_mask_per_position']}")
                print(f"    max_mask_per_position: {scaffold_params['max_mask_per_position']}")
            
            if 'scaffold_operator' not in self.config:
                self.config['scaffold_operator'] = {
                    "fixed_substructure": scaffold_smiles,
                    "mask_mode": mask_mode,
                    "min_masks": scaffold_params['min_masks'],
                    "max_masks": scaffold_params['max_masks'],
                    "min_mask_per_position": scaffold_params['min_mask_per_position'],
                    "max_mask_per_position": scaffold_params['max_mask_per_position'],
                }
            scaffold_handler = initialize_scaffold_handler(self.config, gan_operators, self.device)
        
        # Handle initial population
        data_file = initial_population_file
        if data_file is None:
            if use_scaffold and scaffold_handler is not None:
                if self.verbose:
                    print(f"  Generating initial population from scaffold...")
                data_file = generate_initial_population_from_scaffold(
                    scaffold_handler=scaffold_handler,
                    gan_operators=gan_operators,
                    args=args,
                    log_file=None
                )
            else:
                if self.verbose:
                    print(f"  Generating initial population...")
                
                min_masks = self.config.get('no_scaffold_operator', {}).get('min_masks', 3)
                max_masks = self.config.get('no_scaffold_operator', {}).get('max_masks', 25)
                use_empirical = self.config.get('no_scaffold_operator', {}).get('use_empirical', False)
                dataset_name = dataset_type
                
                data_file = generate_initial_population_without_scaffold(
                    gan_operators=gan_operators,
                    population_size=population_size,
                    batch_size=10,
                    top_k=init_top_k,
                    log_file=None,
                    output_directory=output_dir or "./temp_output",
                    min_masks=min_masks,
                    max_masks=max_masks,
                    use_empirical=use_empirical,
                    dataset_name=dataset_name
                )
        
        # Initialize population
        population = Population(
            gan_operators,
            scoring_operator,
            use_scaffold=use_scaffold,
            scaffold_handler=scaffold_handler,
            mask_mode=mask_mode,
            use_optimized_ppo=True
        )
        
        # Load initial population
        if self.verbose:
            print(f"  Loading population from: {data_file}")
        
        population.read_population_dict_from_file(data_file, population_size)
        all_smiles = set(population.population_sequences)
        
        # Auto-convert chiral molecules to non-chiral for optimization
        original_chiral_count = 0
        if auto_convert_chiral:
            converted_count = 0
            new_sequences = []
            for i, smiles in enumerate(population.population_sequences):
                if has_chirality(smiles):
                    original_chiral_count += 1
                    non_chiral = remove_chirality(smiles)
                    if non_chiral and non_chiral != smiles:
                        if self.verbose:
                            print(f"  转换手性分子: {smiles[:40]}...")
                            print(f"    → {non_chiral[:40]}...")
                        new_sequences.append(non_chiral)
                        converted_count += 1
                    else:
                        new_sequences.append(smiles)
                else:
                    new_sequences.append(smiles)
            
            if converted_count > 0:
                if self.verbose:
                    print(f"  检测到 {original_chiral_count} 个手性分子，已转换为非手性版本")
                population._population_dict[population._data_column_name] = new_sequences
                all_smiles = set(new_sequences)
        
        if self.verbose:
            print(f"  Initial population: {len(all_smiles)} molecules")
        
        if self.verbose:
            print(f"\nRunning optimization for {generations} generations...")
        
        # Set up logging if output directory specified
        log_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = open(f"{output_dir}/run.log", 'w')
            print(f"Logging to {output_dir}/run.log", file=log_file, flush=True)
            print(args, file=log_file, flush=True)
        
        # Train and evolve
        population, all_smiles = train_and_evolve(
            population=population,
            gan_operators=gan_operators,
            args=args,
            train_flags=None,
            all_smiles=all_smiles,
            log_file=log_file,
            scaffold_handler=scaffold_handler,
            use_scaffold=use_scaffold
        )
        
        # Save results if output directory specified
        if output_dir:
            df = self._population_to_dataframe(population)
            df.to_csv(f"{output_dir}/final_population.csv", index=False)
            if log_file:
                log_file.close()
            if self.verbose:
                print(f"  Saved results to: {output_dir}")
                print(f"  Log file: {output_dir}/run.log")
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"  Final population: {population.population_size} molecules")
        
        # Return results
        if return_dataframe:
            return self._population_to_dataframe(population)
        else:
            return population.population_sequences
    
    def _update_config_with_targets(self, target_properties: Dict[str, float], fitness_function=None):
        """Update configuration with target properties.

        Routes each property to the correct predictor:
          - RDKit properties -> scoring_names (computed by RDKit)
          - LiTEN properties  -> scoring_liten_names (predicted by LiTEN-ADMET models)
          - ADMET-AI props   -> scoring_admet_names (predicted by admet_ai package)
        """
        from rlmollm.scoring.property_configs import get_all_properties
        from rlmollm.scoring.tdc_multi_oracle_scoring import TDC_ORACLES

        # RDKit-computed properties (no external model needed)
        rdkit_properties = {'synth', 'drug', 'logP', 'logD', 'number', 'tpsa'}

        # Properties predicted by LiTEN-ADMET models
        # (see LiTEN-ADMET/config/config.json for task_class -> column mapping)
        liten_properties = {
            'Cl_Plasma', 'T12',           # excretion_reg
            'IGC50', 'BCF', 'LC50DM', 'LC50FM',  # toxicity_reg
            'ROA', 'hERG_Blockers', 'hERG_Blockers_10um',
            'Drug_induced_liver_injury', 'AMES_Mutagenicity',
            'FDAMDD', 'Skin_Sensitization', 'Carcinogenicity',
            'Eye_Corrosion', 'Eye_Irritation', 'Respiratory',
            'Human_Hepatotoxicity', 'Drug_induced_Neurotoxicity',
            'Ototoxicity', 'Hematotoxicity', 'Drug_induced_Nephrotoxicity',
            'Genotoxicity', 'RPMI_8226_Immunitoxicity',
            'A549_Cytotoxicity', 'Hek293_Cytotoxicity',
            'logS', 'logD7.4', 'logP', 'Melting_point', 'Boiling_point',
            'pKa_acidic', 'pKa_basic',
            'Caco2_Permeability', 'MDCK_Permeability',
            'PPB', 'VDss', 'Fu',
            'PAMPA', 'Pgp_inhibitor', 'Pgp_substrate', 'HIA',
            'F20%', 'F30%', 'F50%', 'OATP1B1_inhibitor', 'OATP1B3_inhibitor',
            'BCRP_inhibitor', 'BSEP_inhibitor', 'BBB', 'MRP1_inhibitor',
            'CYP1A2_inhibitor', 'CYP1A2_substrate',
            'CYP2C19_inhibitor', 'CYP2C19_substrate',
            'CYP2C9_inhibitor', 'CYP2C9_substrate',
            'CYP2D6_inhibitor', 'CYP2D6_substrate',
            'CYP3A4_inhibitor', 'CYP3A4_substrate',
            'CYP2B6_inhibitor', 'CYP2B6_substrate',
            'CYP2C8_inhibitor', 'HLM_stability',
        }

        available_admet_properties = set(get_all_properties())

        property_name_mapping = {
            'logp': 'logP',
            'logd': 'logD',
        }

        rdkit_props_in_targets = []
        liten_props_in_targets = []
        admet_props_in_targets = []
        tdc_props_in_targets = []
        unknown_props_in_targets = []

        for prop_name in target_properties.keys():
            internal_name = property_name_mapping.get(prop_name, prop_name)
            if internal_name in rdkit_properties:
                rdkit_props_in_targets.append(internal_name)
            elif internal_name in liten_properties:
                liten_props_in_targets.append(internal_name)
            elif internal_name in available_admet_properties:
                admet_props_in_targets.append(internal_name)
            elif internal_name in TDC_ORACLES:
                tdc_props_in_targets.append(internal_name)
            else:
                unknown_props_in_targets.append(internal_name)

        if unknown_props_in_targets:
            print(f"Warning: Unknown properties (no config found): {unknown_props_in_targets}")

        scoring_operator = self.config.get('scoring_operator', {})

        # RDKit-based scoring names (includes 'number' as validity constraint)
        scoring_operator['scoring_names'] = list(set(rdkit_props_in_targets + ['number']))

        # LiTEN external scorer names -> predicted by LiTEN-ADMET models
        scoring_operator['scoring_liten_names'] = list(set(liten_props_in_targets))

        # ADMET-AI names -> predicted by admet_ai package
        scoring_operator['scoring_admet_names'] = list(set(admet_props_in_targets))

        # TDC oracle names -> evaluated by TDC
        scoring_operator['scoring_tdc_names'] = list(set(tdc_props_in_targets))
        
        # Save current TDC oracle name for scoring
        if tdc_props_in_targets:
            self._current_tdc_oracle = tdc_props_in_targets[0]  # Use first TDC oracle
        else:
            self._current_tdc_oracle = None

        # Selection names = TDC oracles for oracle-based optimization
        # This is critical: for TDC oracles, the oracle itself IS the selection criterion
        if tdc_props_in_targets:
            # When using TDC oracles, use them as the ONLY selection criteria
            scoring_operator['selection_names'] = list(set(tdc_props_in_targets))
        else:
            # Otherwise use all target properties
            all_internal_names = [property_name_mapping.get(p, p) for p in target_properties.keys()]
            scoring_operator['selection_names'] = list(set(all_internal_names))

        # Fitness function passed through to scoring operator
        if fitness_function is not None:
            scoring_operator['fitness_function'] = fitness_function

        self.config['scoring_operator'] = scoring_operator

        if self.verbose:
            print(f"  RDKit properties:    {scoring_operator['scoring_names']}")
            print(f"  LiTEN properties:   {scoring_operator['scoring_liten_names']}")
            print(f"  ADMET-AI properties: {scoring_operator['scoring_admet_names']}")
            print(f"  TDC Oracle: {scoring_operator['scoring_tdc_names']}")
            print(f"  Selection properties: {scoring_operator['selection_names']}")
    
    def _calculate_scaffold_params(
        self,
        scaffold_smiles: str,
        min_masks: Optional[int],
        max_masks: Optional[int],
        min_mask_per_position: int,
        max_mask_per_position: int
    ) -> Dict:
        """Calculate scaffold-related mask parameters based on scaffold structure.
        
        This method automatically determines appropriate mask parameters based on:
        - The number of atoms in the scaffold
        - The number of attachment points (#)
        
        Args:
            scaffold_smiles: The scaffold SMILES with # as attachment points
            min_masks: User-specified minimum masks (if None, auto-calculate)
            max_masks: User-specified maximum masks (if None, auto-calculate)
            min_mask_per_position: Minimum masks per attachment point
            max_mask_per_position: Maximum masks per attachment point
            
        Returns:
            Dictionary with calculated parameters
        """
        # Count atoms in scaffold (excluding # attachment points)
        # Also count the # positions
        scaffold_atoms = 0
        attachment_points = 0
        in_brackets = False
        i = 0
        
        while i < len(scaffold_smiles):
            char = scaffold_smiles[i]
            
            # Handle # as attachment point
            if char == '#':
                attachment_points += 1
                i += 1
                continue
            
            # Handle brackets (like [nH], [O-], etc.)
            if char == '[':
                in_brackets = True
                # Count the atom inside brackets
                i += 1
                while i < len(scaffold_smiles) and scaffold_smiles[i] != ']':
                    if scaffold_smiles[i].isalpha():
                        scaffold_atoms += 1
                    i += 1
                if i < len(scaffold_smiles) and scaffold_smiles[i] == ']':
                    in_brackets = False
                i += 1
                continue
            
            # Handle aromatic lowercase letters (count as atoms)
            if char.isalpha() and char.islower():
                scaffold_atoms += 1
            # Handle uppercase letters (start of atom symbols)
            elif char.isalpha() and char.isupper():
                scaffold_atoms += 1
                # Check for two-letter elements (Fe, Cl, Br, etc.)
                if i + 1 < len(scaffold_smiles) and scaffold_smiles[i + 1].islower():
                    i += 1
            
            i += 1
        
        # If we couldn't parse atoms properly, estimate from length
        if scaffold_atoms < 2:
            scaffold_atoms = max(3, len(scaffold_smiles) // 2)
        
        # Auto-calculate min_masks if not provided
        if min_masks is None:
            # min_masks = roughly 1/3 of scaffold atoms, minimum 1
            min_masks = max(1, scaffold_atoms // 3)
        
        # Auto-calculate max_masks if not provided
        if max_masks is None:
            # max_masks = roughly 1x scaffold atoms, with some buffer
            # Also ensure max_masks >= min_masks
            max_masks = max(min_masks + 2, scaffold_atoms + 2)
        
        # Ensure min_masks <= max_masks
        if min_masks > max_masks:
            min_masks = max_masks
        
        # Calculate default min_mask_per_position based on attachment points
        # If there are many attachment points, use smaller per-position masks
        if attachment_points > 0:
            default_per_position = max(1, min_mask_per_position)
        else:
            default_per_position = 1
        
        return {
            'min_masks': min_masks,
            'max_masks': max_masks,
            'min_mask_per_position': default_per_position,
            'max_mask_per_position': max_mask_per_position,
            'scaffold_atoms': scaffold_atoms,
            'attachment_points': attachment_points
        }
    
    def _parse_model_type(self, model_type: str) -> tuple:
        """Parse model type string to get training flags."""
        model_type = model_type.lower()
        
        generator_only = 'alm' in model_type
        mlm_loss = 'alm' in model_type
        use_ppo = 'ppo' in model_type
        no_merge = 'ng' in model_type
        
        return generator_only, mlm_loss, use_ppo, no_merge
    
    def _population_to_dataframe(self, population: Population) -> pd.DataFrame:
        """Convert population to pandas DataFrame with all properties."""
        pop_dict = population.population_dict
        df = pd.DataFrame(pop_dict)
        
        # Reverse mapping: internal names -> user-facing names
        reverse_mapping = {
            'drug': 'qed',
            'drug_raw': 'qed_raw',
            'synth': 'sa',
            'synth_raw': 'sa_raw',
            'logP': 'logp',
            'logP_raw': 'logp_raw',
            'logD': 'logd',
            'logD_raw': 'logd_raw',
        }
        
        df = df.rename(columns=reverse_mapping)
        return df
