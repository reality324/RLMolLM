from rlmollm.scoring.molecule_scoring import MoleculeScoring
from rdkit.Chem import Descriptors
import numpy as np
import scipy.stats
from rdkit.Chem import Crippen

class LinkerScoring(MoleculeScoring):
    """Custom scoring class for linker optimization with target LogP."""

    def __init__(self, style, scoring_names, selection_names, scoring_admet_names=None, scoring_parameters=None, 
                 data_column_name='smiles', fitness_column_name='fitness', 
                 fitness_function=scipy.stats.hmean):
        super().__init__(scoring_names, scoring_admet_names, selection_names, 
                        scoring_parameters=scoring_parameters, 
                        data_column_name=data_column_name, 
                        fitness_column_name=fitness_column_name, 
                        fitness_function=fitness_function)
        
        # Default values for target LogP parameters
        self._target_logp_value = 0.3  # Default target
        self._target_logp_sigma = 1.0  # Default tolerance
        
        # Override with values from scoring_parameters if provided
        if scoring_parameters is not None:
            if 'target_logp_value' in scoring_parameters:
                self._target_logp_value = float(scoring_parameters['target_logp_value'])
            if 'target_logp_sigma' in scoring_parameters:
                self._target_logp_sigma = float(scoring_parameters['target_logp_sigma'])

        # Default values for LogP range
        self._min_logp_value = 2.0
        self._max_logp_value = 3.0
        self._logp_sigma = 0.5  # Controls how quickly score drops outside range

        # Override with values from scoring_parameters if provided
        if scoring_parameters is not None:
            if 'min_logp_value' in scoring_parameters:
                self._min_logp_value = float(scoring_parameters['min_logp_value'])
            if 'max_logp_value' in scoring_parameters:
                self._max_logp_value = float(scoring_parameters['max_logp_value'])
            if 'logp_sigma' in scoring_parameters:
                self._logp_sigma = float(scoring_parameters['logp_sigma'])

        # 1. 添加logD相关参数（仿照logP）
        # target_logD 参数
        self._target_logd_value = 2.0  # 默认目标值（适合CNS药物）
        self._target_logd_sigma = 0.5  # 默认容忍度
        
        # range_logD 参数
        self._min_logd_value = 1.5
        self._max_logd_value = 2.5
        self._logd_sigma = 0.3
        
        # 2. 从scoring_parameters覆盖
        if scoring_parameters is not None:
            # target_logD
            if 'target_logd_value' in scoring_parameters:
                self._target_logd_value = float(scoring_parameters['target_logd_value'])
            if 'target_logd_sigma' in scoring_parameters:
                self._target_logd_sigma = float(scoring_parameters['target_logd_sigma'])
            
            # range_logD
            if 'min_logd_value' in scoring_parameters:
                self._min_logd_value = float(scoring_parameters['min_logd_value'])
            if 'max_logd_value' in scoring_parameters:
                self._max_logd_value = float(scoring_parameters['max_logd_value'])
            if 'logd_sigma' in scoring_parameters:
                self._logd_sigma = float(scoring_parameters['logd_sigma'])

    def get_name_to_function_dict(self):
        """Add custom scoring functions."""
        # Get the base scoring functions
        output_dict = super().get_name_to_function_dict()
        
        # Add our custom LogP scoring function
        output_dict['target_logp'] = self._target_logp_score

        output_dict['range_logp'] = self._range_logp_score

        output_dict['target_logd'] = self._target_logd_score

        output_dict['range_logd'] = self._range_logd_score
        
        return output_dict
    
    def _target_logp_score(self, mol, default=0.0):
        """Score based on LogP proximity to target value."""
        try:
            # logp = Descriptors.MolLogP(mol)
            logp = Crippen.MolLogP(mol)
            # Gaussian score centered at target value
            score = np.exp(-0.5 * ((logp - self._target_logp_value)/self._target_logp_sigma)**2)
            return score
        except:
            return default
        
    def _range_logp_score(self, mol, default=0.0):
        """Score based on LogP falling within a desired range."""
        try:
            # logp = Descriptors.MolLogP(mol)
            logp = Crippen.MolLogP(mol)
            # Get range parameters from scoring_parameters
            min_logp = self._min_logp_value  # e.g., 2.0
            max_logp = self._max_logp_value  # e.g., 3.0
            
            # If LogP is within range, give full score
            if min_logp <= logp <= max_logp:
                return 1.0
            
            # If below minimum, calculate distance from minimum
            elif logp < min_logp:
                distance = min_logp - logp
                # Use sigma for smoothing the transition
                return np.exp(-0.5 * (distance/self._logp_sigma)**2)
            
            # If above maximum, calculate distance from maximum
            else:  # logp > max_logp
                distance = logp - max_logp
                # Use sigma for smoothing the transition
                return np.exp(-0.5 * (distance/self._logp_sigma)**2)
                
        except:
            return default

    def _target_logd_score(self, mol, default=0.0):
        """基于logD接近目标值的评分"""
        try:
            # 使用之前创建的logD计算函数
            logd = Crippen.MolLogD(mol)  # 或者使用已有的_crippen_mol_logD_with_default
            # 高斯评分，中心在目标值
            score = np.exp(-0.5 * ((logd - self._target_logd_value) / 
                                  self._target_logd_sigma) ** 2)
            return score
        except:
            return default
        
    def _range_logd_score(self, mol, default=0.0):
        """基于logD落在理想范围内的评分"""
        try:
            logd = Crippen.MolLogD(mol)
            min_logd = self._min_logd_value
            max_logd = self._max_logd_value
            
            # 如果在范围内，给满分
            if min_logd <= logd <= max_logd:
                return 1.0
            
            # 低于最小值
            elif logd < min_logd:
                distance = min_logd - logd
                return np.exp(-0.5 * (distance / self._logd_sigma) ** 2)
            
            # 高于最大值
            else:
                distance = logd - max_logd
                return np.exp(-0.5 * (distance / self._logd_sigma) ** 2)
                
        except:
            return default