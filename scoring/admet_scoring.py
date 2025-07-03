from scoring.molecule_scoring import MoleculeScoring, remap, mol_to_canonical_smiles
from rdkit.Chem import Descriptors
import numpy as np
import scipy.stats
from rdkit.Chem import Crippen
from admet_ai import ADMETModel
from rdkit.Chem import QED


class ADMETScoring(MoleculeScoring):
    """Custom scoring class for linker optimization with target LogP."""

    def __init__(self, scoring_names, scoring_admet_names, selection_names, property_config, 
                 scoring_parameters=None, 
                 data_column_name='smiles', fitness_column_name='fitness', 
                 fitness_function=scipy.stats.hmean):
        super().__init__(scoring_names, scoring_admet_names, selection_names,
                        scoring_parameters=scoring_parameters, 
                        data_column_name=data_column_name, 
                        fitness_column_name=fitness_column_name, 
                        fitness_function=fitness_function)
        
        self._scoring_admet_names = scoring_admet_names
        self._property_config = property_config

        if scoring_admet_names:
            self.admet_model = ADMETModel()

    def get_name_to_function_dict(self):
        """Get dictionary that maps string to scoring functions.

        Returns:
            Dict[str, function] to relate names to scoring functions
        """
        # Create a new dictionary with our functions
        result_dict = {
            'synth': self._compute_synthetic_accessibility,
            'drug': self._compute_qed,
            'logP': self._compute_logp,
            'number': self._compute_number
        }
        return result_dict
    

    def generate_scores(self, mols):
        """Generate scores for list of rdkit molecules.

        Args:
            mols (List[rdkit.Chem.rdchem.Mol]): List of rdkit molecules

        Returns:
            Dict[str, List[float]] where keys are the scoring function names and the values are the scores
        """
        output = {}

        # Apply scoring functions to molecules
        for scoring_name in self._scoring_names:
            output[scoring_name] = list(map(self._name_to_function[scoring_name], mols))
        
        # Always return data as well
        output[self.data_column_name] = list(map(mol_to_canonical_smiles, mols))

        
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
            
            # Add ADMET scores to output
            for admet_name in self._scoring_admet_names:
                # Extract values as a list and normalize each one
                property_values = admet_df[admet_name].tolist()
                # Only normalize ADMET properties that are used for selection
                if admet_name in self._selection_names:
                    output[admet_name] = [self._normalize_score(val, admet_name) for val in property_values]
                else:
                    # For non-selection properties, just store the raw values
                    output[admet_name] = property_values
        
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
        
    def _normalize_score(self, value, property_name):
        """Normalize a raw property value based on configuration.
        
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
        """Compute synthetic accessibility score normalized according to property config.
        
        Args:
            mol (rdkit.Chem.Mol): Molecule to score
            
        Returns:
            float: Normalized score (0-1)
        """
        try:
            # Compute SA score using the sa_model loaded from the parent class
            if self._sa_model is None:
                return 0.0
                
            score = self._compute_sas(mol)
            
            # Apply normalization based on property config
            return self._normalize_score(score, 'synth')
        except:
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