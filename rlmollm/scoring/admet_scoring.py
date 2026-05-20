from rlmollm.scoring.molecule_scoring import MoleculeScoring, remap, mol_to_canonical_smiles
from rlmollm.scoring.property_configs import merge_property_configs
from rdkit.Chem import Descriptors
import os
import numpy as np
import scipy.stats
from rdkit.Chem import Crippen
from rdkit.Chem import QED


class ADMETScoring(MoleculeScoring):
    """Custom scoring class for linker optimization with target LogP."""

    def __init__(self, scoring_names, scoring_admet_names, selection_names, property_config,
                 scoring_liten_names=None,
                 scoring_tdc_names=None,
                 liten=None,
                 scoring_parameters=None, 
                 data_column_name='smiles', fitness_column_name='fitness', 
                 fitness_function=scipy.stats.hmean,
                 auto_load_properties=True):
        super().__init__(scoring_names, scoring_admet_names, selection_names,
                        scoring_parameters=scoring_parameters, 
                        data_column_name=data_column_name, 
                        fitness_column_name=fitness_column_name, 
                        fitness_function=fitness_function)
                        
        self._scoring_admet_names = scoring_admet_names
        self._scoring_tdc_names = scoring_tdc_names or []
        self._auto_load_properties = auto_load_properties
        
        # Auto-load missing property configurations from property_configs.py
        # Check scoring_names, scoring_admet_names, AND scoring_liten_names
        if auto_load_properties:
            all_props_to_check = list(scoring_names)
            if scoring_admet_names:
                all_props_to_check.extend(scoring_admet_names)
            if scoring_liten_names:
                all_props_to_check.extend(scoring_liten_names)
            
            missing_props = [prop for prop in all_props_to_check if prop not in property_config and prop not in ['number']]
            if missing_props:
                property_config = merge_property_configs(property_config, missing_props)
                print(f"Auto-loaded configurations for: {missing_props}")
        
        self._property_config = property_config
        self._scoring_liten_names = scoring_liten_names or []
        self._liten_config = liten or {}
        self._liten_predictor = None

        # Only import and initialize ADMET-AI if we actually need it (lazy loading)
        if scoring_admet_names:
            try:
                from admet_ai import ADMETModel
                self.admet_model = ADMETModel()
            except ImportError as e:
                print("Warning: admet_ai module not found. ADMET scoring will be unavailable.")
                self.admet_model = None
        else:
            self.admet_model = None

        # Initialize TDC oracles (lazy loading)
        self._tdc_oracles = {}
        self._tdc_initialized = False
    
    def _init_tdc_oracles(self):
        """Initialize TDC oracles for scoring."""
        if self._tdc_initialized:
            return
        
        # RDKit compatibility for TDC
        class _RDKitSixCompat:
            @staticmethod
            def iteritems(d):
                return iter(d.items())
            @staticmethod
            def string_types():
                return (str,)
        
        import sys
        if 'rdkit.six' not in sys.modules:
            sys.modules['rdkit.six'] = _RDKitSixCompat()
        
        import tdc
        
        for oracle_name in self._scoring_tdc_names:
            try:
                self._tdc_oracles[oracle_name] = tdc.Oracle(oracle_name)
                print(f"Initialized TDC oracle: {oracle_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize TDC oracle '{oracle_name}': {e}")
        
        self._tdc_initialized = True
    
    def _normalize_tdc_score(self, score):
        """Normalize TDC oracle score to [0, 1] range.
        
        Most TDC oracles return scores in [0, 1] range or higher.
        This method handles different score ranges appropriately.
        """
        if score < 0:
            return 0.0
        
        # For scores typically in [0, 1] range (most MPO oracles)
        if score <= 1.0:
            return score
        
        # For unbounded positive scores, use sigmoid normalization
        # Score of 0 -> 0.5, higher scores -> closer to 1
        # Use softplus-like normalization
        import math
        if score > 10:
            # Very high scores, use logarithmic scaling
            normalized = 1.0 - 1.0 / (1.0 + math.log1p(score))
            return max(0.0, min(1.0, normalized))
        else:
            # Use sigmoid-like normalization
            normalized = score / (score + 5.0)  # Half-point at score=5
            return max(0.0, min(1.0, normalized))

    def get_name_to_function_dict(self):
        """Get dictionary that maps string to scoring functions.

        Returns:
            Dict[str, function] to relate names to scoring functions
        """
        # Create a new dictionary with our functions
        result_dict = {
            'synth': self._compute_synthetic_accessibility,
            'sa': self._compute_synthetic_accessibility,
            'sa_score': self._compute_synthetic_accessibility,
            'drug': self._compute_qed,
            'qed': self._compute_qed,
            'logP': self._compute_logp,
            "logD": self._compute_logd,
            'number': self._compute_number,
            'tpsa': self._compute_tpsa
        }
        return result_dict
    

    def generate_scores(self, mols):
        """Generate scores for list of rdkit molecules.

        Args:
            mols (List[rdkit.Chem.rdchem.Mol]): List of rdkit molecules

        Returns:
            Dict[str, List[float]] where keys are the scoring function names and the values are the scores
            Also includes raw (unnormalized) values with suffix '_raw'
            Only includes properties that are in selection_names (used for fitness)
        """
        output = {}
        
        # Import SA scorer for clean SA score calculation
        from rdkit.Contrib.SA_Score import sascorer
        
        # Mapping for properties that need raw values
        raw_function_map = {
            'logP': lambda mol: Crippen.MolLogP(mol) if mol else -3.0,
            "logD": lambda mol: self.estimate_logd(mol) if mol else -3.0,
            'synth': lambda mol: sascorer.calculateScore(mol) if mol else 10.0,
            'drug': lambda mol: QED.qed(mol) if mol else 0.0,
            'number': lambda mol: mol.GetNumAtoms() if mol else 0,
            'tpsa': lambda mol: Descriptors.TPSA(mol) if mol else 0.0
        }

        # Apply scoring functions ONLY for properties in selection_names
        # This ensures only properties used for fitness appear in output
        for scoring_name in self._scoring_names:
            # Only include if it's in selection_names (used for fitness)
            if scoring_name in self._selection_names:
                output[scoring_name] = list(map(self._name_to_function[scoring_name], mols))
                # Get raw values if mapping exists
                if scoring_name in raw_function_map:
                    output[scoring_name + '_raw'] = list(map(raw_function_map[scoring_name], mols))
        
        # Always return data as well
        output[self.data_column_name] = list(map(mol_to_canonical_smiles, mols))

        if self._scoring_liten_names:
            smiles_list = output[self.data_column_name]
            try:
                from rlmollm.scoring.liten_admet_scoring import LiTENADMETPredictor
            except Exception as e:
                raise ImportError("Failed to import LiTENADMETPredictor. Ensure LiTEN-ADMET dependencies are installed.") from e

            if self._liten_predictor is None:
                cfg = self._liten_config or {}
                base_path = cfg.get('base_path', '/home/tianwangcong/LiTEN-ADMET')
                ckpt_dir = cfg.get('ckpt_dir', os.path.join(base_path, 'ckpt/best_models_all'))
                conf_json = cfg.get('conf_json', os.path.join(base_path, 'config/config.json'))
                num_confs = cfg.get('num_confs', 3)
                top_k_confs = cfg.get('top_k_confs', 1)
                batch_size = cfg.get('batch_size', 32)
                cache_size = cfg.get('cache_size', 20000)
                device = cfg.get('device', None)

                self._liten_predictor = LiTENADMETPredictor(
                    base_path=base_path,
                    ckpt_dir=ckpt_dir,
                    conf_json=conf_json,
                    device=device,
                    num_confs=num_confs,
                    top_k_confs=top_k_confs,
                    batch_size=batch_size,
                    cache_size=cache_size,
                )

            col_to_task_class = (self._liten_config or {}).get('column_to_task_class', {}) or {}

            _default_task_map = {
                'Cl_Plasma': 'excretion_reg',
                'T12': 'excretion_reg',
                'IGC50': 'toxicity_reg',
                'BCF': 'toxicity_reg',
                'LC50DM': 'toxicity_reg',
                'LC50FM': 'toxicity_reg',
            }
            for name in self._scoring_liten_names:
                # Smart task class selection: user config overrides, then default map, then fallback
                if name in col_to_task_class:
                    task_class = col_to_task_class[name]
                else:
                    task_class = _default_task_map.get(name, 'toxicity_cla')
                df = self._liten_predictor.predict(smiles_list, task_class=task_class, columns=[name])
                if name not in df.columns:
                    output[name + '_raw'] = [0.0 for _ in smiles_list]
                    output[name] = [0.0 for _ in smiles_list]
                    continue

                raw_vals = df[name].astype(float).fillna(0.0).tolist()
                output[name + '_raw'] = raw_vals

                # Use _normalize_score for proper distance/range-based normalization
                # This handles both regression (preferred_value scoring) and classification
                norm_vals = [self._normalize_score(v, name) for v in raw_vals]
                output[name] = norm_vals
        
        if self._scoring_admet_names:
            # Get ADMET predictions for all molecules
            smiles_list = output[self.data_column_name]
            admet_df = self.admet_model.predict(smiles_list)  # Returns a pandas DataFrame
            
            # Validate that all requested ADMET properties exist in the DataFrame
            missing_properties = [prop for prop in self._scoring_admet_names if prop not in admet_df.columns]
            if missing_properties:
                available_cols = list(admet_df.columns)
                raise KeyError(f"The following ADMET properties were not found in prediction results: {missing_properties}. "\
                            f"Available properties: {available_cols[:10]}... (showing first 10 of {len(available_cols)})")
            
            # Add ADMET scores to output (both normalized and raw)
            # Only include ADMET properties that are actually used
            for admet_name in self._scoring_admet_names:
                # Extract raw values as a list
                property_values = admet_df[admet_name].tolist()
                
                # Store raw values
                output[admet_name + '_raw'] = property_values
                
                # Store normalized values
                output[admet_name] = [self._normalize_score(val, admet_name) for val in property_values]
        
        # TDC Oracle scoring (for matching InVirtuoGen's oracle system)
        if self._scoring_tdc_names:
            # Lazy initialization of TDC oracles
            if not self._tdc_initialized:
                self._init_tdc_oracles()
            
            smiles_list = output[self.data_column_name]
            from rdkit import Chem
            
            for oracle_name in self._scoring_tdc_names:
                if oracle_name not in self._tdc_oracles:
                    continue
                
                tdc_oracle = self._tdc_oracles[oracle_name]
                raw_scores = []
                norm_scores = []
                
                for smiles in smiles_list:
                    if smiles is None or smiles == '':
                        raw_scores.append(0.0)
                        norm_scores.append(0.0)
                        continue
                    
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            raw_scores.append(0.0)
                            norm_scores.append(0.0)
                        else:
                            score = tdc_oracle(smiles)
                            if score is None or np.isnan(score):
                                raw_scores.append(0.0)
                                norm_scores.append(0.0)
                            else:
                                raw_scores.append(float(score))
                                # Normalize TDC scores to [0, 1] range
                                # Most TDC oracles return scores in [0, 1] or higher
                                # Use sigmoid-like normalization for unbounded scores
                                norm_scores.append(self._normalize_tdc_score(float(score)))
                    except Exception as e:
                        raw_scores.append(0.0)
                        norm_scores.append(0.0)
                
                output[oracle_name + '_raw'] = raw_scores
                output[oracle_name] = norm_scores
        
        # Calculate fitness scores using both regular and ADMET properties
        output[self._fitness_column_name] = []
        for i in range(len(output[self.data_column_name])):
            # Include both regular selection metrics and ADMET selection metrics
            fitness_array = [output[x][i] for x in self._selection_names]
            
            if len(fitness_array) > 0:
                # Use harmonic mean (or specified fitness function) to combine scores
                output[self._fitness_column_name].append(self._fitness_function(fitness_array))
            else:
                output[self._fitness_column_name].append(-1.0)

        return output

    @property
    def column_names(self):
        base = super().column_names
        extra = []
        if self._scoring_liten_names:
            extra.extend([name + '_raw' for name in self._scoring_liten_names])
            extra.extend(list(self._scoring_liten_names))

        # Preserve order, remove duplicates
        seen = set()
        ordered = []
        for name in list(base) + extra:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered
        
    def _normalize_score(self, value, property_name):
        """Normalize a raw property value based on configuration.
        
        Uses linear distance-based scoring for preferred_value targeting,
        same approach as EvoDiffMol implementation.
        
        For values outside the configured range, uses a softer decay to avoid
        sudden 0 scores. Score never goes below a small epsilon.
        
        Args:
            value (float): The raw property value
            property_name (str): The name of the property for configuration lookup
            
        Returns:
            float: Normalized score between 0 and 1, where 1 is better
        """
        
        config = self._property_config[property_name]
        
        # For percentile values, they're already normalized
        if property_name.endswith('_percentile'):
            return value / 100.0
            
        # Extract configuration values
        value_range = config.get('range', [0, 1])
        x_min, x_max = value_range
        
        # Check for target preferred value (single point target with linear scoring)
        if 'preferred_value' in config:
            target = config['preferred_value']
            
            # Calculate max allowed distance
            max_dist = max(target - x_min, x_max - target)
            dist_from_target = abs(value - target)
            
            if max_dist > 0:
                # Linear score within range
                score = max(0.0, 1.0 - (dist_from_target / max_dist))
                
                # For values outside range, use softer decay instead of 0
                if dist_from_target > max_dist:
                    # Use exponential decay for values outside the main range
                    excess = dist_from_target - max_dist
                    # Soft minimum: ensure score doesn't go below 0.1 for reasonable excess
                    soft_decay = max(0.1, 1.0 - (excess / max_dist) * 0.9)
                    score = min(score, soft_decay)
            else:
                score = 1.0 if value == target else 0.0
            
            return score
        
        # Check for target preferred range
        if 'preferred_range' in config:
            pref_min, pref_max = config['preferred_range']
            
            # Normalize based on how close the value is to the preferred range
            if value < pref_min:
                # Below range - normalize from x_min to pref_min
                normalized = remap(value, x_min, pref_min)
            elif value > pref_max:
                # Above range - normalize from pref_max to x_max
                normalized = 1 - remap(value, pref_max, x_max)
            else:
                # Within preferred range - perfect score
                normalized = 1.0
                
            return max(0.0, min(1.0, normalized))
        
        # For properties without a preferred range, normalize based on directional preference
        else:
            # First convert the value to a 0-1 scale within the specified range
            normalized = remap(value, x_min, x_max)
            
            # Ensure the value stays within 0-1 bounds
            normalized = max(0.0, min(1.0, normalized))
            
            # If higher values are not better (i.e., lower is better),
            # invert the score so that lower raw values get higher scores
            if not config.get('higher_is_better', True):
                normalized = 1.0 - normalized
            
            return normalized
    
    def _compute_synthetic_accessibility(self, mol):
        """Compute synthetic accessibility score using RDKit's SA scorer.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1) according to property config
        """
        try:
            # Use RDKit's built-in SA scorer
            from rdkit.Contrib.SA_Score import sascorer
            score = sascorer.calculateScore(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(score, 'synth')
        except Exception:
            # Return lowest score on failure
            return 0.0

    def _compute_qed(self, mol):
        """Compute QED (drug-likeness) score normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Calculate raw QED score
            raw_score = QED.qed(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(raw_score, 'drug')
        except:
            # Return lowest score on failure
            return 0.0

    def _compute_tpsa(self, mol):
        """Compute TPSA (topological polar surface area) score normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Calculate raw TPSA value
            raw_score = Descriptors.TPSA(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(raw_score, 'tpsa')
        except:
            # Return lowest score on failure
            return 0.0
    def _compute_logd(self, mol):
        """Compute LogD score normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Estimate logD using our simplified calculation
            raw_score = self.estimate_logd(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(raw_score, 'logD')
        except:
            # Return lowest score on failure
            return 0.001  # 返回小的正数，避免调和平均数问题
    def _compute_logp(self, mol):
        """Compute LogP score normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Calculate raw LogP
            raw_score = Crippen.MolLogP(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(raw_score, 'logP')
        except:
            # Return lowest score on failure
            return 0.0

    def estimate_logd(self, mol, pH=7.4):
        """估算logD，确保返回正数"""
        if mol is None:
            return 0.0
        try:
            from rdkit import Chem
            logp = Crippen.MolLogP(mol)
            
            # 简化的估算：考虑分子中可电离基团
            smarts_patterns = {
                'acidic': ['C(=O)[OH]', 'S(=O)(=O)[OH]', 'P(=O)([OH])([OH])'],  # 羧酸、磺酸、磷酸
                'basic': ['[N;H0]', '[N+;H0]', '[N;H1;!$(NC=O)]']  # 胺类
            }
            
            acidic_count = 0
            basic_count = 0
            
            for pattern in smarts_patterns['acidic']:
                acidic_count += len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
            
            for pattern in smarts_patterns['basic']:
                basic_count += len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
            
            # 简化校正：每个可电离基团校正0.3个log单位
            correction = (acidic_count + basic_count) * 0.3
            logd = logp - correction
            
            # 确保返回至少一个小的正数
            return max(logd, -10.0)  # 设置合理的下限
        except:
            return 0.0
        
    def _compute_number(self, mol):
        """Compute number of atoms normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Get number of atoms
            raw_score = mol.GetNumAtoms()
            
            # Apply normalization based on property config
            return self._normalize_score(raw_score, 'number')
        except:
            # Return lowest score on failure
            return 0.0