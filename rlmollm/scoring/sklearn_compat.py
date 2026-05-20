"""
sklearn Compatibility Patch for TDC Oracle Models

TDC's jnk3/gsk3b oracle models were pickled with sklearn 0.23.0.
sklearn 1.8.0 changed the internal _tree dtype to include 'missing_go_to_left'
field, making these models incompatible.

This module patches sklearn at import time to handle both old and new dtypes,
and provides manual tree traversal for prediction.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def apply_sklearn_tree_patch():
    """Apply comprehensive patches to sklearn for old model compatibility."""
    import sklearn
    from sklearn.tree import _tree
    import sklearn.tree._tree as _tree_module

    # Patch 1: Fix _check_node_ndarray for old tree format
    if hasattr(_tree_module, '_check_node_ndarray'):
        original_check = _tree_module._check_node_ndarray

        def patched_check_node_ndarray(nodes, expected_dtype):
            if nodes.dtype.names is not None and 'missing_go_to_left' not in nodes.dtype.names:
                n_nodes = len(nodes)
                new_dtype = np.dtype([
                    ('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'),
                    ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'),
                    ('weighted_n_node_samples', '<f8'), ('missing_go_to_left', 'u1')
                ])
                new_nodes = np.zeros(n_nodes, dtype=new_dtype)
                for name in ['left_child', 'right_child', 'feature', 'threshold',
                             'impurity', 'n_node_samples', 'weighted_n_node_samples']:
                    new_nodes[name] = nodes[name]
                new_nodes['missing_go_to_left'] = 0
                return original_check(new_nodes, expected_dtype)
            return original_check(nodes, expected_dtype)

        _tree_module._check_node_ndarray = patched_check_node_ndarray

    # Patch 2: Add missing monotonic_cst attribute to old RandomForest models
    try:
        from sklearn.ensemble._forest import RandomForestClassifier
        if not hasattr(RandomForestClassifier, 'monotonic_cst'):
            RandomForestClassifier.monotonic_cst = None
    except ImportError:
        pass

    # Patch 3: Add missing attributes to old RandomForest models on load
    original_setattr = object.__setattr__ if hasattr(object, '__setattr__') else None
    
    # Patch 4: Fix sklearn_tags for old estimators
    try:
        from sklearn.base import BaseEstimator
        _original_get_tags = BaseEstimator.__sklearn_tags__
        
        def patched_get_tags(self):
            try:
                return _original_get_tags(self)
            except (AttributeError, TypeError, KeyError):
                from sklearn.utils._tags import Tags
                return Tags(
                    input_tags=True,
                    target_tags=False,
                    paired_tags=False,
                    sparse_tags=False,
                    no_validation_tags=False,
                    allow_nan_tags=True,
                    poor_score_tags=False,
                    requires_positive_X_tags=False,
                    allows_zero_samples=False,
                    _skip_sanity_check=True,
                )
        BaseEstimator.__sklearn_tags__ = patched_get_tags
    except Exception as e:
        pass  # Patch may not be needed or available in this sklearn version

    return True


def monkey_patch_sklearn_validation():
    """Patch sklearn estimator validation to work with old-style models."""
    from sklearn.utils.validation import _is_unsupported_attribute
    import sklearn.utils.validation as validation_module

    original = validation_module._is_unsupported_attribute

    def patched(attr, estimator):
        try:
            return original(attr, estimator)
        except (AttributeError, TypeError):
            return True

    validation_module._is_unsupported_attribute = patched


class PatchedRandomForestClassifier:
    """
    Wrapper around RandomForestClassifier that bypasses sklearn 1.8 validation
    for old 0.23.0 pickled models and uses raw tree data for prediction.
    """

    def __init__(self, model, n_features=2048):
        self._model = model
        self._n_features = n_features

    def predict_proba(self, X):
        """Predict probabilities by averaging individual tree predictions."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, 2), dtype=np.float64)

        for tree in self._model.estimators_:
            tree_preds = self._predict_single_tree(tree, X)
            predictions += tree_preds

        predictions /= len(self._model.estimators_)
        return predictions

    def _predict_single_tree(self, tree, X):
        """Manually traverse tree using raw arrays for prediction."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, 2), dtype=np.float64)

        feature = np.asarray(tree.tree_.feature, dtype=np.int32)
        threshold = np.asarray(tree.tree_.threshold, dtype=np.float64)
        left_child = np.asarray(tree.tree_.children_left, dtype=np.int32)
        right_child = np.asarray(tree.tree_.children_right, dtype=np.int32)
        values = np.asarray(tree.tree_.value[:, 0, :], dtype=np.float64)

        # Validate feature indices are within valid range
        valid_features = (feature >= -2) & (feature < self._n_features)
        if not np.all(valid_features):
            invalid_count = np.sum(~valid_features)
            raise ValueError(
                f"Invalid feature indices detected: {invalid_count} values out of range. "
                f"Expected [-2, {self._n_features}), got min={feature.min()}, max={feature.max()}"
            )

        for i in range(n_samples):
            ni = 0
            max_iterations = 10000  # Prevent infinite loops
            iterations = 0
            
            while True:
                fi = feature[ni]
                
                # Safety check: prevent infinite loop
                iterations += 1
                if iterations > max_iterations:
                    raise RuntimeError(
                        f"Infinite loop detected at sample {i}: "
                        f"feature[{ni}]={fi}, threshold[{ni}]={threshold[ni]:.4f}, "
                        f"left_child={left_child[ni]}, right_child={right_child[ni]}, "
                        f"X[{i}, {fi}]={X[i, fi] if 0 <= fi < self._n_features else 'N/A'}"
                    )
                
                # Leaf node check (feature < 0 indicates leaf)
                if fi < 0:
                    break
                
                # Traverse tree
                if X[i, fi] <= threshold[ni]:
                    ni = left_child[ni]
                else:
                    ni = right_child[ni]

            leaf = values[ni]
            total = leaf[0] + leaf[1]
            if total > 0:
                predictions[i, 0] = leaf[0] / total
                predictions[i, 1] = leaf[1] / total
            else:
                predictions[i, 0] = 0.5
                predictions[i, 1] = 0.5

        return predictions


def create_tdc_compat_oracle(oracle_name, model_path):
    """Create a working TDC oracle with sklearn compatibility patches.
    
    This function wraps old sklearn models in a PatchedRandomForestClassifier
    that bypasses sklearn's internal validation to work with models from sklearn 0.23.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    class WorkingOracle:
        def __init__(self, model):
            # Wrap the model in our patched classifier
            self._model = PatchedRandomForestClassifier(model, n_features=2048)

        def __call__(self, smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            features = np.zeros((1, 2048), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp, features[0])
            proba = self._model.predict_proba(features)
            return float(proba[0, 1])

    import pickle
    
    # Apply patch before loading
    apply_sklearn_tree_patch()
    
    # Load the old model
    model = pickle.load(open(model_path, 'rb'))
    
    # Wrap in our compatibility layer
    wrapped_model = WorkingOracle(model)
    return wrapped_model
