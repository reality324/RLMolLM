import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
import random
import torch
import json
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem import DataStructs
from rdkit.Contrib.SA_Score import sascorer # Added for SA Score calculation
from tqdm import tqdm # For progress bar during calculation
import umap # Added for UMAP visualization
import re

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # Set deterministic behavior for CuDNN if available
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Try to seed RDKit if possible
    try:
        Chem.SetRandomSeed(seed)
    except:
        print(f"Warning: Could not set RDKit random seed. Some operations might not be fully reproducible.")
    # For UMAP reproducibility
    umap.UMAP(random_state=seed)

def get_method_color_mapping():
    """
    Get a consistent color mapping for all methods across all plots.
    
    Returns:
        Dictionary mapping method names to colors
    """
    # Define the color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    # Define consistent method order and color mapping
    method_color_map = {
        "LM": "#1f77b4",          # Blue
        "LM-NG": "#ff7f0e",       # Orange  
        "LM-RL": "#2ca02c",       # Green
        "ALM": "#d62728",         # Red
        "ALM-RL": "#9467bd",      # Purple
        "LM/LM-NG": "#ff7f0e",    # Orange (same as LM-NG for combined case)
        "LM-NG-RL": "#8c564b",    # Brown
        "Fixed-LM": "#e377c2",    # Pink
        "Fixed-LM-NG": "#7f7f7f", # Gray
        "Initial": "black",       # Black for initial population
        "initial": "black",       # Handle lowercase variant
        # Internal lowercase names for compatibility
        "lm": "#1f77b4",          # Blue (same as LM)
        "lm-ng": "#ff7f0e",       # Orange (same as LM-NG)
        "lm-rl": "#2ca02c",       # Green (same as LM-RL)
        "alm": "#d62728",         # Red (same as ALM)
        "alm-rl": "#9467bd",      # Purple (same as ALM-RL)
    }
    
    return method_color_map

# Function to calculate validity and uniqueness of molecules
def calculate_mol_stats(filepath):
    """
    Calculates validity percentage and uniqueness percentage from SMILES file.
    
    Args:
        filepath: Path to CSV/TSV file containing 'smiles' column
    
    Returns:
        Dictionary with validity and uniqueness percentages
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return {'validity': 0.0, 'uniqueness': 0.0}
    
    try:
        if filepath.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t', engine='python')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=',')
        else:
            print(f"Error: Unsupported file format for {filepath}")
            return {'validity': 0.0, 'uniqueness': 0.0}
        
        if 'smiles' in df.columns:
            print(f"Calculating stats for {filepath}...")
            
            # Check validity and track unique valid SMILES
            valid_count = 0
            total_count = len(df)
            unique_smiles = set()
            
            for smi in tqdm(df['smiles'], desc="Checking molecules"):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_count += 1
                    # Get canonical SMILES to ensure consistency in uniqueness check
                    canonical_smi = Chem.MolToSmiles(mol)
                    unique_smiles.add(canonical_smi)
            
            # Calculate percentages
            valid_percent = (valid_count / total_count) * 100 if total_count > 0 else 0
            unique_percent = (len(unique_smiles) / valid_count) * 100 if valid_count > 0 else 0
            
            print(f"Validity: {valid_count}/{total_count} molecules ({valid_percent:.2f}%)")
            print(f"Uniqueness: {len(unique_smiles)}/{valid_count} molecules ({unique_percent:.2f}%)")
            
            return {
                'validity': valid_percent,
                'uniqueness': unique_percent
            }
        else:
            print(f"Warning: 'smiles' column not found in {filepath}")
            return {'validity': 0.0, 'uniqueness': 0.0}
    except Exception as e:
        print(f"Error processing {filepath} for molecule stats: {e}")
        return {'validity': 0.0, 'uniqueness': 0.0}

# Function to calculate properties (with option to force recalculation for specific properties)
def calculate_properties(df, force_recalculate=None):
    """
    Calculates logP, QED (drug), and SA Score (synth) if they are missing and 'smiles' exists.
    
    Args:
        df: DataFrame with molecule data
        force_recalculate: List of property names to recalculate even if they already exist
    """
    # Map internal names to RDKit functions
    property_funcs = {
        "logP": Descriptors.MolLogP,
        "drug": QED.qed,
        "synth": sascorer.calculateScore
    }
    
    # Determine properties to calculate
    if force_recalculate is None:
        force_recalculate = []
    
    # Properties missing from DataFrame OR explicitly requested to recalculate
    to_calculate = list(set(
        [prop for prop in property_funcs if prop not in df.columns] + 
        [prop for prop in force_recalculate if prop in property_funcs]
    ))
    
    if 'smiles' in df.columns and to_calculate:
        props_to_calculate_str = ", ".join(to_calculate)
        forced_str = " (including forced recalculation)" if any(p in force_recalculate for p in to_calculate) else ""
        print(f"Calculating properties: {props_to_calculate_str}{forced_str}")
        
        # Pre-convert SMILES to Mol objects
        mols = []
        valid_indices = []
        for i, smi in enumerate(df['smiles']):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(i)
        
        # Initialize or reset columns for calculation
        for prop_name in to_calculate:
            df[prop_name] = pd.Series(index=df.index, dtype=float)
        
        # Calculate properties
        for prop_name in tqdm(to_calculate, desc="Calculating properties"):
            calc_func = property_funcs[prop_name]
            calculated_values = []
            for mol in mols:
                val = None
                try:
                    val = calc_func(mol)
                except Exception as e:
                    pass
                calculated_values.append(val)
            
            # Assign calculated values
            if calculated_values:
                df.loc[df.index[valid_indices], prop_name] = calculated_values
    
    return df

def load_data(filepath):
    """Loads data from CSV or TSV file and calculates properties if needed."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        if filepath.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t', engine='python')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=',')
        else:
            print(f"Error: Unsupported file format for {filepath}")
            return None

        # Define required columns
        required_cols = ['logP', 'drug', 'synth']
        
        # Calculate missing properties
        print(f"Processing file: {filepath}")
        
        # ALWAYS recalculate the 'synth' and 'logP' properties for consistency
        df = calculate_properties(df, force_recalculate=['synth', 'logP'])
        
        # Check if required columns exist after calculation
        if not all(col in df.columns for col in required_cols):
            missing_in_file = [col for col in required_cols if col not in df.columns]
            print(f"Warning: Following columns still missing: {missing_in_file}")
        
        return df
    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None

# NEW: Function to compute Morgan fingerprints for SMILES
def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    """
    Compute the Morgan fingerprint for a SMILES string.
    
    Args:
        smiles (str): A SMILES representation of a molecule.
        radius (int): Radius for the Morgan fingerprint.
        nBits (int): Size of the fingerprint.
        
    Returns:
        np.array or None: The fingerprint as a numpy array if successful, else None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# NEW: Function to compute all fingerprints for a list of SMILES
def compute_fingerprints_for_dataframes(data_dict):
    """
    Compute Morgan fingerprints for all molecules in the data dictionary.
    
    Args:
        data_dict: Dictionary of DataFrames with SMILES data
        
    Returns:
        Dictionary with method names as keys and fingerprints as values
    """
    fps_dict = {}
    
    for method_name, df in data_dict.items():
        if df is None or 'smiles' not in df.columns:
            continue
            
        print(f"Computing fingerprints for {method_name}...")
        fps = []
        valid_indices = []
        
        for i, smi in enumerate(tqdm(df['smiles'], desc=f"{method_name} fingerprints")):
            fp = smiles_to_fingerprint(smi)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(i)
                
        # Only keep fingerprints with corresponding property data
        if fps:
            fps_dict[method_name] = {
                'fps': np.array(fps),
                'indices': valid_indices,
                'df': df.iloc[valid_indices].reset_index(drop=True)
            }
            print(f"  Computed {len(fps)} fingerprints for {method_name}")
            
    return fps_dict

# NEW: Function to plot UMAP visualizations
def plot_umap_visualization(fps_dict, properties, output_dir="plots", random_state=42, umap_params=None):
    """
    Generates UMAP visualizations for the methods with one subplot per method in one row.
    
    Args:
        fps_dict: Dictionary with fingerprints for each method
        properties: List of properties to visualize
        output_dir: Directory to save the plots
        random_state: Random state for UMAP
        umap_params: Optional dictionary with UMAP parameters
    """
    if not fps_dict:
        print("No fingerprint data available for UMAP visualization")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define ordered method names
    ordered_methods = ["lm", "lm-ng", "lm-rl", "alm", "alm-rl"]
    # Filter to methods actually present in the data
    methods_to_plot = [m for m in ordered_methods if m in fps_dict]
    
    if not methods_to_plot:
        print("No methods available for UMAP visualization")
        return
    
    # Map internal property names to descriptive labels for plots
    property_labels = {
        'logP': 'logP',
        'drug': 'QED',
        'synth': 'SA'
    }
    
    # Configure color maps and ranges for each property
    property_configs = {
        "logP": {"cmap": "viridis", "vmin": -2, "vmax": 6},
        "drug": {"cmap": "viridis", "vmin": 0, "vmax": 1},
        "synth": {"cmap": "plasma_r", "vmin": 1, "vmax": 7}  # Using reversed plasma for SA (lower is better)
    }
    
    # Use provided UMAP parameters or defaults
    default_umap_params = {
        "n_neighbors": 5,
        "min_dist": 0.1,
        "metric": "jaccard"
    }
    
    # Merge user-provided parameters with defaults
    umap_config = default_umap_params.copy()
    if umap_params:
        umap_config.update(umap_params)
    
    print(f"UMAP configuration for visualization: {umap_config}")
    
    # Increase font sizes for better readability - much larger fonts
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18
    })
    
    # For each property, create a separate figure with all methods in one row
    for prop in properties:
        if prop not in property_labels:
            continue
            
        prop_label = property_labels[prop]
        prop_config = property_configs.get(prop, {"cmap": "viridis", "vmin": None, "vmax": None})
            
        print(f"\nGenerating UMAP visualization for {prop} ({prop_label})")
        
        # Create a smaller figure with one row and multiple columns (one per method)
        fig, axes = plt.subplots(1, len(methods_to_plot), figsize=(3 * len(methods_to_plot), 3))
        if len(methods_to_plot) == 1:
            axes = [axes]  # Make sure axes is a list for consistent indexing
        
        # Process each method
        for i, method_name in enumerate(methods_to_plot):
            if method_name not in fps_dict:
                continue
                
            method_data = fps_dict[method_name]
            fps = method_data['fps']
            df = method_data['df']
            
            if prop not in df.columns:
                print(f"  Property {prop} not available for {method_name}, skipping")
                continue
                
            property_values = df[prop].values
            
            # Apply UMAP to fingerprints
            print(f"  Computing UMAP for {method_name}...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_config["n_neighbors"],
                min_dist=umap_config["min_dist"],
                metric=umap_config["metric"],
                random_state=random_state
            )
            embedding = reducer.fit_transform(fps)
            
            # Plot on the corresponding axis
            ax = axes[i]
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=property_values, 
                cmap=prop_config["cmap"],
                vmin=prop_config["vmin"],
                vmax=prop_config["vmax"],
                s=150,  # Even larger dots for better visibility
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )
            
            # No title - removed as requested
            ax.set_xlabel("UMAP 1", fontsize=22)
            if i == 0:  # Only add y-label to the first subplot
                ax.set_ylabel("UMAP 2", fontsize=22)
                
            # Set integer tick labels for UMAP axes
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
            # Add grid for better readability
            ax.grid(linestyle='--', alpha=0.3)
                
            # Add colorbar with improved formatting
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(prop_label, fontsize=22)
            cbar.ax.tick_params(labelsize=20)
        
        # No overall title - removed as requested
        
        # Adjust layout and save figure
        plt.tight_layout()  # No need for extra space since no suptitle
        plt.savefig(os.path.join(output_dir, f"umap_{prop}_methods.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All UMAP visualizations completed")

# NEW: Function to create combined UMAP visualization with both epoch 1 and best epoch models
def plot_combined_epochs_umap(fps_dict_best, fps_dict_epoch1, output_dir="plots", random_state=42, umap_params=None):
    """
    Create a single UMAP plot with all methods from both epoch 1 and best epoch,
    coloring by the method and using different marker styles for different epochs.
    
    Args:
        fps_dict_best: Dictionary with fingerprints for best epoch methods
        fps_dict_epoch1: Dictionary with fingerprints for epoch 1 methods
        output_dir: Directory to save the plots
        random_state: Random state for UMAP
        umap_params: Optional dictionary with UMAP parameters
    """
    if not fps_dict_best and not fps_dict_epoch1:
        print("No fingerprint data available for combined epochs UMAP visualization")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating combined UMAP visualization for all methods across both epochs...")
    
    # Define ordered method names
    ordered_methods = ["lm", "lm-ng", "lm-rl", "alm", "alm-rl"]
    
    # Collect all available methods across both epochs
    all_methods = set()
    if fps_dict_best:
        all_methods.update(m for m in ordered_methods if m in fps_dict_best)
    if fps_dict_epoch1:
        all_methods.update(m for m in ordered_methods if m in fps_dict_epoch1)
    
    # Filter to methods present and order them appropriately
    methods_to_plot = [m for m in ordered_methods if m in all_methods]
    
    if not methods_to_plot:
        print("No methods available for combined epochs UMAP visualization")
        return
        
    # Use centralized color mapping for consistency
    all_method_colors = get_method_color_mapping()
    method_colors = {method: all_method_colors.get(method, "#808080") for method in methods_to_plot}
    
    # Combine all fingerprints into a single dataset with method and epoch labels
    all_fps = []
    method_labels = []
    epoch_labels = []
    
    # Add best epoch data
    for method in methods_to_plot:
        if fps_dict_best and method in fps_dict_best:
            method_data = fps_dict_best[method]
            method_fps = method_data['fps']
            
            all_fps.append(method_fps)
            method_labels.extend([method] * len(method_fps))
            epoch_labels.extend(["best"] * len(method_fps))
    
    # Add epoch 1 data
    for method in methods_to_plot:
        if fps_dict_epoch1 and method in fps_dict_epoch1:
            method_data = fps_dict_epoch1[method]
            method_fps = method_data['fps']
            
            all_fps.append(method_fps)
            method_labels.extend([method] * len(method_fps))
            epoch_labels.extend(["epoch1"] * len(method_fps))
    
    # Concatenate all fingerprints
    combined_fps = np.vstack(all_fps)
    
    # Use provided UMAP parameters or defaults
    default_umap_params = {
        "n_neighbors": 5,
        "min_dist": 0.1,
        "metric": "jaccard"
    }
    
    # Merge user-provided parameters with defaults
    umap_config = default_umap_params.copy()
    if umap_params:
        umap_config.update(umap_params)
    
    print(f"UMAP configuration for combined epochs plot: {umap_config}")
    
    # Apply UMAP to combined fingerprints
    print("Computing UMAP embedding for combined epoch data...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_config["n_neighbors"],
        min_dist=umap_config["min_dist"],
        metric=umap_config["metric"],
        random_state=random_state
    )
    embedding = reducer.fit_transform(combined_fps)
    
    # Create the plot - smaller figure size
    plt.figure(figsize=(10, 8))
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18
    })
    
    # Define marker styles for different epochs - UPDATED for better visibility
    epoch_markers = {
        "best": "o",     # circle for best epoch (will be empty)
        "epoch1": "*"    # star for epoch 1 (will be filled)
    }
    
    # Plot each method with its own color, and use markers to distinguish epochs
    for method in methods_to_plot:
        # Plot best epoch data for this method
        if fps_dict_best and method in fps_dict_best:
            indices = [i for i, (m, e) in enumerate(zip(method_labels, epoch_labels)) 
                       if m == method and e == "best"]
            
            if indices:
                plt.scatter(
                    embedding[indices, 0], 
                    embedding[indices, 1],
                    facecolors='none',  # Make empty circles
                    edgecolors=method_colors[method],
                    marker=epoch_markers["best"],
                    linewidth=1.5,
                    label=f"{method} (best)",
                    s=150,  # Much larger size for better visibility
                    alpha=0.8
                )
        
        # Plot epoch 1 data for this method
        if fps_dict_epoch1 and method in fps_dict_epoch1:
            indices = [i for i, (m, e) in enumerate(zip(method_labels, epoch_labels)) 
                       if m == method and e == "epoch1"]
            
            if indices:
                plt.scatter(
                    embedding[indices, 0], 
                    embedding[indices, 1],
                    c=method_colors[method],
                    marker=epoch_markers["epoch1"],
                    label=f"{method} (epoch 1)",
                    s=200,  # Extra large size for stars
                    alpha=0.8
                )
    
    # Add labels and legend - no title as requested
    plt.xlabel("UMAP 1", fontsize=22)
    plt.ylabel("UMAP 2", fontsize=22)
    
    # Set integer tick labels for UMAP axes
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Create a more organized legend
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=3, markerscale=1.5)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_combined_methods.png"), dpi=300)
    plt.close()
    
    print("Combined epochs UMAP visualization completed")

def plot_combined_methods_umap(fps_dict, output_dir="plots", random_state=42, umap_params=None):
    """
    Create a single UMAP plot with all methods combined, coloring by the method that generated each molecule.
    
    Args:
        fps_dict: Dictionary with fingerprints for each method
        output_dir: Directory to save the plots
        random_state: Random state for UMAP
        umap_params: Optional dictionary with UMAP parameters
    """
    if not fps_dict:
        print("No fingerprint data available for combined methods UMAP visualization")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating combined UMAP visualization for all methods...")
    
    # Define ordered method names - add initial at the end if present
    ordered_methods = list(fps_dict.keys())
    
    # Remove "initial" or "Initial" if present, to add it at the end
    for initial_variant in ["initial", "Initial"]:
        if initial_variant in ordered_methods:
            ordered_methods.remove(initial_variant)
            # Always use proper capitalization for initial population
            ordered_methods.append("Initial")
        
    # Filter to methods actually present in the data
    methods_to_plot = ordered_methods
    
    if not methods_to_plot:
        print("No methods available for UMAP visualization")
        return
        
    # Use centralized color mapping for consistency
    all_method_colors = get_method_color_mapping()
    method_colors = {method: all_method_colors.get(method, "#808080") for method in methods_to_plot}
    
    # Override color for initial population to black (already handled in centralized mapping)
    if "Initial" in method_colors:
        method_colors["Initial"] = "black"
    
    # Define different marker shapes for different methods
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'P', 'X']
    method_markers = {method: markers[i % len(markers)] for i, method in enumerate(methods_to_plot)}
    
    # Use a special marker for initial population
    if "Initial" in methods_to_plot:
        method_markers["Initial"] = '*'  # Use star for initial population
    
    # Combine all fingerprints into a single dataset with method labels
    all_fps = []
    method_labels = []
    
    for method in methods_to_plot:
        # Get the key to use for fps_dict lookup
        fps_key = method
        # Special case for "Initial" - check if "initial" exists in fps_dict
        if method == "Initial" and "Initial" not in fps_dict and "initial" in fps_dict:
            fps_key = "initial"
            
        if fps_key in fps_dict:
            method_data = fps_dict[fps_key]
            method_fps = method_data['fps']
            
            all_fps.append(method_fps)
            method_labels.extend([method] * len(method_fps))
    
    # Concatenate all fingerprints
    combined_fps = np.vstack(all_fps)
    
    # Use provided UMAP parameters or defaults
    default_umap_params = {
        "n_neighbors": 5,
        "min_dist": 0.1,
        "metric": "jaccard"
    }
    
    # Merge user-provided parameters with defaults
    umap_config = default_umap_params.copy()
    if umap_params:
        umap_config.update(umap_params)
    
    print(f"UMAP configuration for combined methods plot: {umap_config}")
    
    # Apply UMAP to combined fingerprints
    print("Computing UMAP embedding for combined methods data...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_config["n_neighbors"],
        min_dist=umap_config["min_dist"],
        metric=umap_config["metric"],
        random_state=random_state
    )
    embedding = reducer.fit_transform(combined_fps)
    
    # Create the plot - square figure size
    plt.figure(figsize=(8, 8))
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18
    })
    
    # Plot each method with its own color and marker
    for method in methods_to_plot:
        # Get indices for this method
        indices = [i for i, label in enumerate(method_labels) if label == method]
        
        # Plot with larger markers and method-specific shapes
        if method == "Initial":
            # Special case for initial population - larger markers
            plt.scatter(
                embedding[indices, 0], 
                embedding[indices, 1],
                c=method_colors[method],
                marker=method_markers[method],
                label=method,
                s=200,  # Larger stars for initial population
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )
        else:
            plt.scatter(
                embedding[indices, 0], 
                embedding[indices, 1],
                c=method_colors[method],
                marker=method_markers[method],
                label=method,
                s=150,  # Large markers for better visibility
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5
            )
    
    # Add labels and legend - no title as requested
    plt.xlabel("UMAP 1", fontsize=22)
    plt.ylabel("UMAP 2", fontsize=22)
    
    # Set integer tick labels for UMAP axes
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Create a more organized legend
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=3, markerscale=1.5)
    
    # Add grid for better readability
    plt.grid(linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    # Save as umap_methods.png
    plt.savefig(os.path.join(output_dir, "umap_methods.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Combined methods UMAP visualization completed")

def plot_distributions(data_dict, properties, mol_stats=None, output_dir="plots", logp_lines=None):
    """Plots distributions for specified properties from multiple data sources on horizontal subplots."""
    print(f"\nPlot distributions called with logp_lines={logp_lines}")
    
    num_properties = len(properties)
    
    # Add plots for molecule statistics (validity, uniqueness)
    num_mol_stats = 0
    if mol_stats and 'validity' in mol_stats[next(iter(mol_stats))]:
        num_mol_stats += 1  # Validity plot
    if mol_stats and 'uniqueness' in mol_stats[next(iter(mol_stats))]:
        num_mol_stats += 1  # Uniqueness plot
    
    total_plots = num_properties + num_mol_stats
    
    # Set font sizes for property distribution plots
    plt.rcParams.update({
        'font.size': 10,        # Smaller default font size
        'axes.titlesize': 12,   # Smaller title size
        'axes.labelsize': 11,   # Smaller axis label size
        'xtick.labelsize': 9,   # Smaller x-tick label size
        'ytick.labelsize': 9,   # Smaller y-tick label size
        'legend.fontsize': 10   # Smaller legend font size
    })
    
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5), sharey=False)

    if total_plots == 1: # Handle case with a single property
        axes = [axes]

    # Map internal property names to descriptive labels for plots
    property_labels = {
        'logP': 'logP',
        'drug': 'QED',      # Use QED label for plot
        'synth': 'SA'       # Rename from "SA Score" to just "SA" for plot
    }

    # Get available keys - these should already be the display names
    available_keys = list(data_dict.keys()) if data_dict else []
    if mol_stats:
        available_keys = list(set(available_keys + list(mol_stats.keys())))
    
    # Create an ordered list of keys based on expected display name order
    # Use the keys present in the data, but in a specific order with Initial at the end
    expected_order = ["LM/LM-NG", "LM-NG-RL", "LM-RL", "ALM", "ALM-RL", "Fixed-LM", "Fixed-LM-NG"]
    ordered_available_keys = []
    
    # First add keys in the expected order (if they exist in the data)
    for key in expected_order:
        if key in available_keys:
            ordered_available_keys.append(key)
    
    # Then add any remaining keys (except 'initial' / 'Initial')
    for key in available_keys:
        if key.lower() != 'initial' and key not in ordered_available_keys:
            ordered_available_keys.append(key)
    
    # Add 'Initial' at the end if any variation of it exists
    if 'initial' in available_keys:
        ordered_available_keys.append('Initial')
    elif 'Initial' in available_keys:
        ordered_available_keys.append('Initial')

    # Use centralized color mapping for consistency
    all_method_colors = get_method_color_mapping()
    file_color_map = {name: all_method_colors.get(name, "#808080") for name in ordered_available_keys}
    
    # Set initial population to black if it exists
    if "Initial" in file_color_map:
        file_color_map["Initial"] = "black"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plot_index = 0
    
    # First plot molecule statistics if provided
    if mol_stats:
        # Plot validity
        if 'validity' in mol_stats[next(iter(mol_stats))]:
            ax = axes[plot_index]
            
            # Get data for bar chart in specified order - exclude 'initial'/'Initial'
            names = []
            values = []
            errors = []
            bar_colors = []
            
            for name in ordered_available_keys:
                # Skip 'initial' for validity plot
                if name.lower() == "initial":
                    continue
                
                # Get the key to use for mol_stats lookup (could be different case)
                stats_key = name
                if name not in mol_stats:
                    # Try finding a case-insensitive match
                    matching_keys = [k for k in mol_stats.keys() if k.lower() == name.lower()]
                    if matching_keys:
                        stats_key = matching_keys[0]
                    else:
                        continue
                
                if 'validity' in mol_stats[stats_key]:
                    names.append(name)
                    values.append(mol_stats[stats_key]['validity'] / 100.0)  # Convert to 0-1 scale
                    
                    # Get error value if available (from multiple samples)
                    if 'validity_std' in mol_stats[stats_key]:
                        errors.append(mol_stats[stats_key]['validity_std'] / 100.0)  # Convert to 0-1 scale
                    else:
                        errors.append(0)  # No error if only one sample
                        
                    bar_colors.append(file_color_map[name])
            
            # Create bar chart with error bars
            bars = ax.bar(names, values, yerr=errors, color=bar_colors, 
                          edgecolor='black', linewidth=1, capsize=5, error_kw={'elinewidth': 1.5})
            
            ax.set_ylabel('Validity', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1)  # Set y-axis from 0 to 1 for proportion
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars as proportions
            for j, v in enumerate(values):
                # Position the text higher if there's an error bar
                offset = 0.02 if errors[j] < 0.02 else errors[j] + 0.01
                ax.text(j, v + offset, f"{v:.3f}", ha='center', fontsize=9)
            
            # Print the standard deviations for reference
            if any(e > 0 for e in errors):
                for j, (name, value, error) in enumerate(zip(names, values, errors)):
                    print(f"  {name} validity: {value:.3f} ± {error:.3f}")
            
            plot_index += 1
        
        # Plot uniqueness
        if 'uniqueness' in mol_stats[next(iter(mol_stats))]:
            ax = axes[plot_index]
            
            # Get data for bar chart in specified order - exclude 'initial'/'Initial'
            names = []
            values = []
            errors = []
            bar_colors = []
            
            for name in ordered_available_keys:
                # Skip 'initial' for uniqueness plot
                if name.lower() == "initial":
                    continue
                
                # Get the key to use for mol_stats lookup (could be different case)
                stats_key = name
                if name not in mol_stats:
                    # Try finding a case-insensitive match
                    matching_keys = [k for k in mol_stats.keys() if k.lower() == name.lower()]
                    if matching_keys:
                        stats_key = matching_keys[0]
                    else:
                        continue
                    
                if 'uniqueness' in mol_stats[stats_key]:
                    names.append(name)
                    values.append(mol_stats[stats_key]['uniqueness'] / 100.0)  # Convert to 0-1 scale
                    
                    # Get error value if available (from multiple samples)
                    if 'uniqueness_std' in mol_stats[stats_key]:
                        errors.append(mol_stats[stats_key]['uniqueness_std'] / 100.0)  # Convert to 0-1 scale
                    else:
                        errors.append(0)  # No error if only one sample
                        
                    bar_colors.append(file_color_map[name])
            
            # Create bar chart with error bars
            bars = ax.bar(names, values, yerr=errors, color=bar_colors, 
                          edgecolor='black', linewidth=1, capsize=5, error_kw={'elinewidth': 1.5})
            
            ax.set_ylabel('Uniqueness', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1)  # Set y-axis from 0 to 1 for proportion
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars as proportions
            for j, v in enumerate(values):
                # Position the text higher if there's an error bar
                offset = 0.02 if errors[j] < 0.02 else errors[j] + 0.01
                ax.text(j, v + offset, f"{v:.3f}", ha='center', fontsize=9)
            
            # Print the standard deviations for reference
            if any(e > 0 for e in errors):
                for j, (name, value, error) in enumerate(zip(names, values, errors)):
                    print(f"  {name} uniqueness: {value:.3f} ± {error:.3f}")
            
            plot_index += 1
    
    # Then plot the other properties
    for i, prop in enumerate(properties):
        ax = axes[plot_index]
        plotted_something = False
        
        # Use the mapped label for printing and plot titles/labels
        plot_label = property_labels.get(prop, prop) # Default to internal name if not mapped
        print(f"\n--- Plotting: {prop} (as {plot_label}) ---")
        
        # Create a dictionary to store data for each method in the ordered list
        ordered_data = {}
        
        # First collect all data in the specified order
        for name in ordered_available_keys:
            # Get the key to use for data_dict lookup (could be different case)
            data_key = name
            if name.lower() == "initial" and name not in data_dict:
                # Try to find any variation of "initial" in the keys
                initial_keys = [k for k in data_dict.keys() if k.lower() == "initial"]
                if initial_keys:
                    data_key = initial_keys[0]
            
            if data_key in data_dict and data_dict[data_key] is not None and prop in data_dict[data_key].columns:
                valid_data = data_dict[data_key][prop].dropna()
                if not valid_data.empty:
                    ordered_data[name] = valid_data
                    print(f"  Plotting {name} (valid data points: {len(valid_data)})" )
                else:
                    print(f"  Skipping {name} for {prop}: No valid data (all NaN?).")
            elif data_key in data_dict and data_dict[data_key] is not None:
                print(f"  Skipping {name} for {prop}: Column not found.")
        
        # Then plot the data in the specified order
        for name, valid_data in ordered_data.items():
            # Use just lines without fill for clearer visualization
            sns.kdeplot(valid_data, ax=ax, label=name, color=file_color_map[name], 
                       fill=False, linewidth=2)
            plotted_something = True

        if plotted_something:
            # Remove title
            ax.set_xlabel(plot_label, fontweight='bold', fontsize=11) # Use mapped label
            # Set y-label to 'Density' for all property plots
            ax.set_ylabel('Density', fontweight='bold', fontsize=11)
            
            # Add grid for better readability
            ax.grid(linestyle='--', alpha=0.7)
            
            # Set y-limit specifically for logP plot
            if prop == 'logP': # Check internal name
                ax.set_ylim(0, 1) 
                print(f"  Setting y-axis limit to (0, 1) for {prop}")
                
                # Add vertical lines for logP values from the config if provided
                if logp_lines and isinstance(logp_lines, list):
                    print(f"  Using custom logP lines from config: {logp_lines}")
                    for logp_value in logp_lines:
                        ax.axvline(x=logp_value, color='black', linestyle='--', linewidth=1.5)
                        print(f"  Added vertical dashed line at logP = {logp_value}")
                else:
                    # Default vertical lines at logP = 2 and logP = 3 if no config provided
                    print(f"  No custom logP lines found (logp_lines={logp_lines}), using defaults")
                    ax.axvline(x=2, color='black', linestyle='--', linewidth=1.5)
                    ax.axvline(x=3, color='black', linestyle='--', linewidth=1.5)
                    print(f"  Added default vertical dashed lines at logP = 2 and logP = 3")
            
            # Set x-limit specifically for QED plot to enforce 0-1 range
            if prop == 'drug': # Check internal name
                ax.set_xlim(0, 1) # QED range from 0 to 1
                print(f"  Setting x-axis fixed range to (0, 1) for {prop} ({plot_label})")
            
            # Set x-limit specifically for SA plot to show standard 1-7 scale
            if prop == 'synth': # Check internal name
                ax.set_xlim(1, 7) # SA range from 1 to 7
                print(f"  Setting x-axis fixed range to (1, 7) for {prop} ({plot_label})")
            
            # Create a more visible and contrasting legend with the ordered names
            handles, labels = ax.get_legend_handles_labels()
            
            # Create a mapping from labels to handles
            label_to_handle = dict(zip(labels, handles))
            
            # Get the ordered existing labels
            ordered_labels = [label for label in ordered_available_keys if label in labels]
            ordered_handles = [label_to_handle[label] for label in ordered_labels]
            
            leg = ax.legend(ordered_handles, ordered_labels, frameon=True, fancybox=True, shadow=True, fontsize=10)
            leg.get_frame().set_edgecolor('black')
        else:
            print(f"No data plotted for {prop} (as {plot_label}). Skipping axis setup.")
            # Optionally hide the empty subplot
            # ax.set_visible(False)
        
        plot_index += 1

    plt.tight_layout()
    output_filename = os.path.join(output_dir, "property_distributions.png")
    plt.savefig(output_filename, dpi=300)  # Increased DPI for better quality
    print(f"\nCombined plot saved to {output_filename}")
    plt.close() # Close the figure
    
    # Reset matplotlib defaults to avoid affecting other plots
    plt.rcdefaults()

def plot_molecular_properties_only(data_dict, properties, output_dir="plots", logp_lines=None):
    """Plots distributions for only molecular properties (SA, QED, logP) excluding validity and uniqueness."""
    print(f"\nPlot molecular properties only called with logp_lines={logp_lines}")
    
    # Filter to only molecular properties (exclude validity/uniqueness)
    molecular_props = [prop for prop in properties if prop in ['synth', 'drug', 'logP']]
    
    if not molecular_props:
        print("No molecular properties found to plot")
        return
    
    num_properties = len(molecular_props)
    
    # Set font sizes for property distribution plots - larger fonts for better readability
    plt.rcParams.update({
        'font.size': 14,        # Larger default font size
        'axes.titlesize': 18,   # Larger title size
        'axes.labelsize': 16,   # Larger axis label size
        'xtick.labelsize': 14,  # Larger x-tick label size
        'ytick.labelsize': 14,  # Larger y-tick label size
        'legend.fontsize': 14   # Larger legend font size
    })
    
    fig, axes = plt.subplots(1, num_properties, figsize=(5 * num_properties, 5), sharey=False)

    if num_properties == 1: # Handle case with a single property
        axes = [axes]

    # Map internal property names to descriptive labels for plots
    property_labels = {
        'logP': 'logP',
        'drug': 'QED',      # Use QED label for plot
        'synth': 'SA'       # Rename from "SA Score" to just "SA" for plot
    }

    # Get available keys - these should already be the display names
    available_keys = list(data_dict.keys()) if data_dict else []
    
    # Create an ordered list of keys based on expected display name order
    # Use the keys present in the data, but in a specific order with Initial at the end
    expected_order = ["LM/LM-NG", "LM-NG-RL", "LM-RL", "ALM", "ALM-RL", "Fixed-LM", "Fixed-LM-NG"]
    ordered_available_keys = []
    
    # First add keys in the expected order (if they exist in the data)
    for key in expected_order:
        if key in available_keys:
            ordered_available_keys.append(key)
    
    # Then add any remaining keys (except 'initial' / 'Initial')
    for key in available_keys:
        if key.lower() != 'initial' and key not in ordered_available_keys:
            ordered_available_keys.append(key)
    
    # Add 'Initial' at the end if any variation of it exists
    if 'initial' in available_keys:
        ordered_available_keys.append('Initial')
    elif 'Initial' in available_keys:
        ordered_available_keys.append('Initial')

    # Use centralized color mapping for consistency
    all_method_colors = get_method_color_mapping()
    file_color_map = {name: all_method_colors.get(name, "#808080") for name in ordered_available_keys}
    
    # Set initial population to black if it exists
    if "Initial" in file_color_map:
        file_color_map["Initial"] = "black"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot molecular properties only
    for i, prop in enumerate(molecular_props):
        ax = axes[i]
        plotted_something = False
        
        # Use the mapped label for printing and plot titles/labels
        plot_label = property_labels.get(prop, prop) # Default to internal name if not mapped
        print(f"\n--- Plotting: {prop} (as {plot_label}) ---")
        
        # Create a dictionary to store data for each method in the ordered list
        ordered_data = {}
        
        # First collect all data in the specified order
        for name in ordered_available_keys:
            # Get the key to use for data_dict lookup (could be different case)
            data_key = name
            if name.lower() == "initial" and name not in data_dict:
                # Try to find any variation of "initial" in the keys
                initial_keys = [k for k in data_dict.keys() if k.lower() == "initial"]
                if initial_keys:
                    data_key = initial_keys[0]
            
            if data_key in data_dict and data_dict[data_key] is not None and prop in data_dict[data_key].columns:
                valid_data = data_dict[data_key][prop].dropna()
                if not valid_data.empty:
                    ordered_data[name] = valid_data
                    print(f"  Plotting {name} (valid data points: {len(valid_data)})" )
                else:
                    print(f"  Skipping {name} for {prop}: No valid data (all NaN?).")
            elif data_key in data_dict and data_dict[data_key] is not None:
                print(f"  Skipping {name} for {prop}: Column not found.")
        
        # Then plot the data in the specified order
        for name, valid_data in ordered_data.items():
            # Use just lines without fill for clearer visualization
            sns.kdeplot(valid_data, ax=ax, label=name, color=file_color_map[name], 
                       fill=False, linewidth=2)
            plotted_something = True

        if plotted_something:
            # Remove title
            ax.set_xlabel(plot_label, fontweight='bold', fontsize=16) # Use mapped label
            # Set y-label to 'Density' for all property plots
            ax.set_ylabel('Density', fontweight='bold', fontsize=16)
            
            # Add grid for better readability
            ax.grid(linestyle='--', alpha=0.7)
            
            # Set y-limit specifically for logP plot
            if prop == 'logP': # Check internal name
                ax.set_ylim(0, 1) 
                print(f"  Setting y-axis limit to (0, 1) for {prop}")
                
                # Add vertical lines for logP values from the config if provided
                if logp_lines and isinstance(logp_lines, list):
                    print(f"  Using custom logP lines from config: {logp_lines}")
                    for logp_value in logp_lines:
                        ax.axvline(x=logp_value, color='black', linestyle='--', linewidth=1.5)
                        print(f"  Added vertical dashed line at logP = {logp_value}")
                else:
                    # Default vertical lines at logP = 2 and logP = 3 if no config provided
                    print(f"  No custom logP lines found (logp_lines={logp_lines}), using defaults")
                    ax.axvline(x=2, color='black', linestyle='--', linewidth=1.5)
                    ax.axvline(x=3, color='black', linestyle='--', linewidth=1.5)
                    print(f"  Added default vertical dashed lines at logP = 2 and logP = 3")
            
            # Set x-limit specifically for QED plot to enforce 0-1 range
            if prop == 'drug': # Check internal name
                ax.set_xlim(0, 1) # QED range from 0 to 1
                print(f"  Setting x-axis fixed range to (0, 1) for {prop} ({plot_label})")
            
            # Set x-limit specifically for SA plot to show standard 1-7 scale
            if prop == 'synth': # Check internal name
                ax.set_xlim(1, 7) # SA range from 1 to 7
                print(f"  Setting x-axis fixed range to (1, 7) for {prop} ({plot_label})")
            
            # Create a more visible and contrasting legend with the ordered names
            handles, labels = ax.get_legend_handles_labels()
            
            # Create a mapping from labels to handles
            label_to_handle = dict(zip(labels, handles))
            
            # Get the ordered existing labels
            ordered_labels = [label for label in ordered_available_keys if label in labels]
            ordered_handles = [label_to_handle[label] for label in ordered_labels]
            
            leg = ax.legend(ordered_handles, ordered_labels, frameon=True, fancybox=True, shadow=True, fontsize=14)
            leg.get_frame().set_edgecolor('black')
        else:
            print(f"No data plotted for {prop} (as {plot_label}). Skipping axis setup.")

    plt.tight_layout()
    output_filename = os.path.join(output_dir, "molecular_properties_only.png")
    plt.savefig(output_filename, dpi=300)  # Increased DPI for better quality
    print(f"\nMolecular properties only plot saved to {output_filename}")
    plt.close() # Close the figure
    
    # Reset matplotlib defaults to avoid affecting other plots
    plt.rcdefaults()

def calculate_fitness_scores(data_dict):
    """
    Calculate fitness scores for each method using harmonic mean of properties.
    
    Args:
        data_dict: Dictionary of DataFrames with molecular property data
        
    Returns:
        Dictionary with fitness statistics for each method
    """
    fitness_stats = {}
    
    for method_name, df in data_dict.items():
        if df is None or df.empty:
            continue
            
        # Skip initial population
        if method_name.lower() == 'initial':
            continue
            
        # Properties for fitness calculation
        required_props = ['synth', 'drug', 'logP']
        
        # Check if all required properties are available
        if not all(prop in df.columns for prop in required_props):
            print(f"Warning: Missing properties for fitness calculation in {method_name}")
            continue
        
        # Calculate fitness for each molecule
        fitness_scores = []
        
        for _, row in df.iterrows():
            # Get property values
            sa_score = row['synth']  # Synthetic accessibility (1-7, lower is better)
            qed_score = row['drug']  # Drug-likeness (0-1, higher is better)
            logp_value = row['logP']  # Lipophilicity
            
            # Skip if any property is NaN
            if pd.isna(sa_score) or pd.isna(qed_score) or pd.isna(logp_value):
                continue
            
            # Normalize properties to 0-1 scale (higher is better)
            # SA Score: Transform so that lower SA (easier synthesis) gives higher score
            sa_normalized = max(0, (7 - sa_score) / 6)  # Transform 1-7 to 1-0, then to 0-1
            
            # QED is already 0-1, higher is better
            qed_normalized = max(0, min(1, qed_score))
            
            # LogP: Optimal range is typically 1-3 for drug-like molecules
            # Use a penalty function that gives maximum score at logP = 2
            if logp_value <= 2:
                logp_normalized = max(0, logp_value / 2)  # Linear increase from 0 to 1
            else:
                logp_normalized = max(0, 1 - (logp_value - 2) / 4)  # Linear decrease, 0 at logP = 6
            
            logp_normalized = max(0, min(1, logp_normalized))
            
            # Calculate harmonic mean of normalized properties
            properties = [sa_normalized, qed_normalized, logp_normalized]
            if all(p > 0 for p in properties):
                fitness = len(properties) / sum(1/p for p in properties)
            else:
                fitness = 0  # If any property is 0, fitness is 0
            
            fitness_scores.append(fitness)
        
        # Calculate statistics
        if fitness_scores:
            fitness_mean = np.mean(fitness_scores)
            fitness_std = np.std(fitness_scores) if len(fitness_scores) > 1 else 0
            
            fitness_stats[method_name] = {
                'fitness_mean': fitness_mean,
                'fitness_std': fitness_std,
                'fitness_scores': fitness_scores,
                'num_molecules': len(fitness_scores)
            }
            
            print(f"{method_name} fitness: {fitness_mean:.3f} ± {fitness_std:.3f} (n={len(fitness_scores)})")
        else:
            print(f"No valid fitness scores calculated for {method_name}")
    
    return fitness_stats

def save_performance_tables(mol_stats, fitness_stats, output_dir):
    """
    Save performance metrics (validity, uniqueness, fitness) to CSV and formatted text tables.
    
    Args:
        mol_stats: Dictionary with validity and uniqueness statistics
        fitness_stats: Dictionary with fitness statistics
        output_dir: Directory to save the tables
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all method names from both dictionaries
    all_methods = set()
    if mol_stats:
        all_methods.update(mol_stats.keys())
    if fitness_stats:
        all_methods.update(fitness_stats.keys())
    
    # Remove 'initial' or 'Initial' from method ordering for main table
    methods_to_include = [m for m in all_methods if m.lower() != 'initial']
    
    # Create a comprehensive performance table
    performance_data = []
    
    for method in sorted(methods_to_include):
        row = {'Method': method}
        
        # Add validity metrics
        if mol_stats and method in mol_stats and 'validity' in mol_stats[method]:
            validity_mean = mol_stats[method]['validity']
            validity_std = mol_stats[method].get('validity_std', 0)
            row['Validity (%)'] = f"{validity_mean:.1f}"
            if validity_std > 0:
                row['Validity_Std'] = f"{validity_std:.1f}"
        else:
            row['Validity (%)'] = "N/A"
            row['Validity_Std'] = "N/A"
        
        # Add uniqueness metrics
        if mol_stats and method in mol_stats and 'uniqueness' in mol_stats[method]:
            uniqueness_mean = mol_stats[method]['uniqueness']
            uniqueness_std = mol_stats[method].get('uniqueness_std', 0)
            row['Uniqueness (%)'] = f"{uniqueness_mean:.1f}"
            if uniqueness_std > 0:
                row['Uniqueness_Std'] = f"{uniqueness_std:.1f}"
        else:
            row['Uniqueness (%)'] = "N/A"
            row['Uniqueness_Std'] = "N/A"
        
        # Add fitness metrics
        if fitness_stats and method in fitness_stats:
            fitness_mean = fitness_stats[method].get('fitness_mean', 0)
            fitness_std = fitness_stats[method].get('fitness_std', 0)
            row['Fitness Score'] = f"{fitness_mean:.3f}"
            if fitness_std > 0:
                row['Fitness_Std'] = f"{fitness_std:.3f}"
        else:
            row['Fitness Score'] = "N/A"
            row['Fitness_Std'] = "N/A"
        
        performance_data.append(row)
    
    # Create DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "performance_metrics.csv")
    performance_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to: {csv_path}")
    
    # Create a formatted text table for easy reading
    txt_path = os.path.join(output_dir, "performance_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("Performance Metrics Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Create a nicely formatted table
        f.write(f"{'Method':<12} {'Validity (%)':<12} {'Uniqueness (%)':<15} {'Fitness Score':<15}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in performance_df.iterrows():
            method = row['Method']
            validity = row['Validity (%)']
            uniqueness = row['Uniqueness (%)']
            fitness = row['Fitness Score']
            
            # Add error bars if available
            validity_std = row.get('Validity_Std', 'N/A')
            uniqueness_std = row.get('Uniqueness_Std', 'N/A')
            fitness_std = row.get('Fitness_Std', 'N/A')
            
            if validity_std != 'N/A' and validity_std != '0.0':
                validity_display = f"{validity}±{validity_std}"
            else:
                validity_display = validity
                
            if uniqueness_std != 'N/A' and uniqueness_std != '0.0':
                uniqueness_display = f"{uniqueness}±{uniqueness_std}"
            else:
                uniqueness_display = uniqueness
                
            if fitness_std != 'N/A' and fitness_std != '0.0':
                fitness_display = f"{fitness}±{fitness_std}"
            else:
                fitness_display = fitness
            
            f.write(f"{method:<12} {validity_display:<12} {uniqueness_display:<15} {fitness_display:<15}\n")
        
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Validity: Percentage of generated molecules that are chemically valid\n")
        f.write("- Uniqueness: Percentage of valid molecules that are unique (non-duplicated)\n")
        f.write("- Fitness Score: Harmonic mean of normalized molecular properties\n")
        f.write("- Error bars (±) shown when multiple samples are available\n")
    
    print(f"Formatted performance table saved to: {txt_path}")
    
    # Create LaTeX table format
    latex_path = os.path.join(output_dir, "performance_metrics_latex.txt")
    with open(latex_path, 'w') as f:
        f.write("% LaTeX table for performance metrics\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\caption{Performance metrics for molecular generation}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Validity (\\%) & Uniqueness (\\%) & Fitness Score \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in performance_df.iterrows():
            method = row['Method']
            validity = row['Validity (%)']
            uniqueness = row['Uniqueness (%)']
            fitness = row['Fitness Score']
            
            f.write(f"{method} & {validity} & {uniqueness} & {fitness} \\\\\n")
        
        f.write("\\botrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_path}")

def save_molecules_with_properties_and_fitness(data_dict, output_dir):
    """
    Save all molecules with their properties and fitness scores to CSV files.
    
    Args:
        data_dict: Dictionary of DataFrames with molecular property data
        output_dir: Directory to save the CSV files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, df in data_dict.items():
        if df is None or df.empty:
            continue
            
        # Generate CSV for all methods including initial population
        # Removed skip condition to allow initial population CSV output
            
        # Properties for fitness calculation
        required_props = ['synth', 'drug', 'logP']
        
        # Check if all required properties are available
        if not all(prop in df.columns for prop in required_props):
            print(f"Warning: Missing properties for fitness calculation in {method_name}")
            continue
        
        # Create a copy of the dataframe to add fitness scores
        output_df = df.copy()
        
        # Calculate fitness for each molecule
        fitness_scores = []
        
        for _, row in output_df.iterrows():
            # Get property values
            sa_score = row['synth']  # Synthetic accessibility (1-7, lower is better)
            qed_score = row['drug']  # Drug-likeness (0-1, higher is better)
            logp_value = row['logP']  # Lipophilicity
            
            # Skip if any property is NaN
            if pd.isna(sa_score) or pd.isna(qed_score) or pd.isna(logp_value):
                fitness_scores.append(np.nan)
                continue
            
            # Normalize properties to 0-1 scale (higher is better)
            # SA Score: Transform so that lower SA (easier synthesis) gives higher score
            sa_normalized = max(0, (7 - sa_score) / 6)  # Transform 1-7 to 1-0, then to 0-1
            
            # QED is already 0-1, higher is better
            qed_normalized = max(0, min(1, qed_score))
            
            # LogP: Optimal range is typically 1-3 for drug-like molecules
            # Use a penalty function that gives maximum score at logP = 2
            if logp_value <= 2:
                logp_normalized = max(0, logp_value / 2)  # Linear increase from 0 to 1
            else:
                logp_normalized = max(0, 1 - (logp_value - 2) / 4)  # Linear decrease, 0 at logP = 6
            
            logp_normalized = max(0, min(1, logp_normalized))
            
            # Calculate harmonic mean of normalized properties
            properties = [sa_normalized, qed_normalized, logp_normalized]
            if all(p > 0 for p in properties):
                fitness = len(properties) / sum(1/p for p in properties)
            else:
                fitness = 0  # If any property is 0, fitness is 0
            
            fitness_scores.append(fitness)
        
        # Add fitness scores to the dataframe
        output_df['fitness'] = fitness_scores
        
        # Filter out molecules with NaN fitness scores (invalid molecules)
        valid_df = output_df.dropna(subset=['fitness'])
        
        # Remove duplicates based on SMILES to ensure uniqueness
        if 'smiles' in valid_df.columns:
            # Get canonical SMILES for proper duplicate detection
            canonical_smiles = []
            valid_indices = []
            
            for idx, smi in valid_df['smiles'].items():
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    canonical_smi = Chem.MolToSmiles(mol)
                    canonical_smiles.append(canonical_smi)
                    valid_indices.append(idx)
            
            # Create a new dataframe with canonical SMILES
            if canonical_smiles:
                canonical_df = valid_df.loc[valid_indices].copy()
                canonical_df['canonical_smiles'] = canonical_smiles
                
                # Remove duplicates based on canonical SMILES
                unique_df = canonical_df.drop_duplicates(subset=['canonical_smiles'])
                
                # Drop the canonical_smiles column as it's just for deduplication
                unique_df = unique_df.drop(columns=['canonical_smiles'])
                
                # Sort by fitness score (highest first)
                unique_df = unique_df.sort_values('fitness', ascending=False)
                
                # Reorder columns to put SMILES first, then properties, then fitness
                column_order = ['smiles']
                # Add property columns in a logical order
                property_cols = ['logP', 'drug', 'synth']  # logP, QED, SA
                for col in property_cols:
                    if col in unique_df.columns:
                        column_order.append(col)
                
                # Add fitness column
                column_order.append('fitness')
                
                # Add any remaining columns
                remaining_cols = [col for col in unique_df.columns if col not in column_order]
                column_order.extend(remaining_cols)
                
                # Reorder the dataframe
                unique_df = unique_df[column_order]
                
                # Save to CSV
                csv_filename = f"{method_name.lower().replace('/', '_').replace('-', '_')}_molecules_with_fitness.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                unique_df.to_csv(csv_path, index=False)
                
                print(f"Saved {len(unique_df)} valid unique molecules with properties and fitness for {method_name} to: {csv_path}")
                print(f"  Columns: {list(unique_df.columns)}")
                print(f"  Fitness range: {unique_df['fitness'].min():.3f} - {unique_df['fitness'].max():.3f}")
            else:
                print(f"No valid molecules found for {method_name}")
        else:
            print(f"No 'smiles' column found for {method_name}")

def format_metric_with_std(value, std_value):
    """
    Helper function to format metrics with standard deviation, handling nan values properly.
    
    Args:
        value: The main metric value
        std_value: The standard deviation value
        
    Returns:
        Formatted string for display
    """
    if std_value == 'N/A' or std_value == '0.0' or 'nan' in str(std_value).lower():
        return str(value)
    else:
        return f"{value}±{std_value}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze and plot property distributions from molecule generation outputs.")
    parser.add_argument("--epoch1", action="store_true", help="Analyze epoch 1 models instead of best models")
    parser.add_argument("--combined_epochs", action="store_true", help="Create combined plots with both epoch 1 and best models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--with_initial", action="store_true", help="Include initial population in analysis")
    parser.add_argument("--initial_molecules", type=str, default="./output/500_initial/initial_population.csv", 
                       help="Path to initial population file")
    parser.add_argument("--mapping_file", type=str, help="File containing mapping from file prefix to display name")
    # Add new arguments for directory paths
    parser.add_argument("--best_input_dir", type=str, default="output_inference_ns", 
                       help="Directory containing best epoch model data")
    parser.add_argument("--best_output_dir", type=str, default="analysis/plots_ns", 
                       help="Directory to save best epoch plots")
    parser.add_argument("--epoch1_input_dir", type=str, default="output_inference_ns_e1", 
                       help="Directory containing epoch 1 model data")
    parser.add_argument("--epoch1_output_dir", type=str, default="analysis/plots_ns_e1", 
                       help="Directory to save epoch 1 plots")
    # Add new arguments for configuration
    parser.add_argument("--config_file", type=str, help="Path to configuration file for scaffold properties")
    # Add new arguments for UMAP parameters
    parser.add_argument("--umap_neighbors", type=int, default=5, 
                       help="Number of neighbors for UMAP (higher values: more global structure)")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, 
                       help="Minimum distance for UMAP (lower values: tighter clusters)")
    parser.add_argument("--umap_metric", type=str, default="jaccard", 
                       help="Distance metric for UMAP (e.g., 'euclidean', 'cosine', 'jaccard')")
    # Add new argument for number of samples per method
    parser.add_argument("--num_samples", type=int, default=1, 
                       help="Number of samples per method (for error bars)")
    args = parser.parse_args()
    
    # Load mapping from file prefix to display name if provided
    prefix_to_display = {}
    display_to_prefix = {}
    if args.mapping_file and os.path.exists(args.mapping_file):
        print(f"Loading display name mapping from: {args.mapping_file}")
        with open(args.mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    # Format is "Display Name: file_prefix"
                    display, prefix = [part.strip() for part in line.split(':', 1)]
                    display_to_prefix[display] = prefix
                    prefix_to_display[prefix] = display
                    print(f"  Mapping: {display} → {prefix}")
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed} for reproducible analysis")
    print(f"Number of samples per method: {args.num_samples}")
    
    # Extract logP preferred range directly from config file if provided
    config_logp_range = None
    if args.config_file and os.path.exists(args.config_file):
        print(f"Reading config from: {args.config_file}")
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
                
            # Check for scoring_operator.property_config.logP.preferred_range
            if ("scoring_operator" in config and 
                "property_config" in config["scoring_operator"] and 
                "logP" in config["scoring_operator"]["property_config"] and
                "preferred_range" in config["scoring_operator"]["property_config"]["logP"]):
                
                preferred_range = config["scoring_operator"]["property_config"]["logP"]["preferred_range"]
                if isinstance(preferred_range, list) and len(preferred_range) == 2:
                    config_logp_range = [float(preferred_range[0]), float(preferred_range[1])]
                    print(f"Using logP preferred range from config: {config_logp_range}")
        except Exception as e:
            print(f"Error reading config file: {e}")
    
    # Store UMAP parameters for passing to visualization functions
    umap_params = {
        "n_neighbors": args.umap_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": args.umap_metric
    }
    print(f"Using UMAP parameters: {umap_params}")
    
    # For combined epochs visualization
    if args.combined_epochs:
        print("Running combined epochs analysis mode")
        
        # Process best epoch data
        best_input_dir = args.best_input_dir
        best_output_dir = args.best_output_dir
        print(f"Loading best epoch data from {best_input_dir}")
        
        # Load best epoch data
        best_data, best_fps_dict, _ = process_data(best_input_dir, best_output_dir, args.seed, 
                                                 plot_results=False, label="best epoch", 
                                                 prefix_to_display=prefix_to_display,
                                                 display_to_prefix=display_to_prefix,
                                                 logp_range=config_logp_range,
                                                 umap_params=umap_params,
                                                 num_samples=args.num_samples,
                                                 initial_molecules_file=args.initial_molecules)
        
        # Process epoch 1 data
        epoch1_input_dir = args.epoch1_input_dir
        epoch1_output_dir = args.epoch1_output_dir
        print(f"Loading epoch 1 data from {epoch1_input_dir}")
        
        # Load epoch 1 data
        epoch1_data, epoch1_fps_dict, _ = process_data(epoch1_input_dir, epoch1_output_dir, args.seed, 
                                                     plot_results=False, label="epoch 1",
                                                     prefix_to_display=prefix_to_display,
                                                     display_to_prefix=display_to_prefix,
                                                     logp_range=config_logp_range,
                                                     umap_params=umap_params,
                                                     num_samples=args.num_samples,
                                                     initial_molecules_file=args.initial_molecules)
        
        # Create combined visualization with both epochs
        if best_fps_dict and epoch1_fps_dict:
            plot_combined_epochs_umap(best_fps_dict, epoch1_fps_dict, 
                                     output_dir=best_output_dir, 
                                     random_state=args.seed,
                                     umap_params=umap_params)
        else:
            print("Error: Could not create combined plot as data is missing for one or both epochs")
            
        return
        
    # Standard single-epoch analysis
    input_dir = args.best_input_dir
    output_dir = args.best_output_dir
    
    if args.epoch1:
        input_dir = args.epoch1_input_dir
        output_dir = args.epoch1_output_dir
        print(f"Analyzing epoch 1 models from {input_dir}")
    
    # Run standard analysis
    data, fps_dict, mol_stats = process_data(input_dir, output_dir, args.seed, 
                                           plot_results=True,
                                           prefix_to_display=prefix_to_display,
                                           display_to_prefix=display_to_prefix,
                                           logp_range=config_logp_range,
                                           umap_params=umap_params,
                                           num_samples=args.num_samples,
                                           initial_molecules_file=args.initial_molecules)
    
    # Process initial population if requested
    if args.with_initial:
        print(f"\nProcessing initial population from {args.initial_molecules}")
        
        # Check if file exists
        if not os.path.exists(args.initial_molecules):
            print(f"Error: Initial population file not found at {args.initial_molecules}")
            print("Please provide a valid file path using --initial_molecules.")
            return
        
        # Create a dictionary with just the initial population
        initial_data = {}
        
        # Load and process the initial population
        initial_df = load_data(args.initial_molecules)
        if initial_df is not None:
            initial_data["initial"] = initial_df
            
            # Calculate fingerprints for the initial population
            initial_fps_dict = compute_fingerprints_for_dataframes(initial_data)
            
            if initial_fps_dict and "initial" in initial_fps_dict:
                # Add initial population to the existing fps_dict
                fps_dict["initial"] = initial_fps_dict["initial"]
                
                # Add initial population data to the existing data for property distributions
                data["initial"] = initial_data["initial"]
                
                # Now replot the distributions with the initial population included
                properties_to_plot = ['synth', 'drug', 'logP']
                print("\nGenerating property distributions with initial population included...")
                # Include mol_stats to show validity and uniqueness - Initial population won't be included in these stats
                plot_distributions(data, properties_to_plot, mol_stats=mol_stats, output_dir=output_dir, logp_lines=config_logp_range)
                
                # Plot molecular properties only (SA, QED, logP) without validity/uniqueness
                print("\nGenerating molecular properties only plot with initial population included...")
                plot_molecular_properties_only(data, properties_to_plot, output_dir=output_dir, logp_lines=config_logp_range)
                
                # Generate UMAP visualization with the initial population included
                print("\nGenerating UMAP visualization with initial population included...")
                plot_umap_visualization(fps_dict, properties_to_plot, output_dir=output_dir, random_state=args.seed, umap_params=umap_params)
            else:
                print("Error: Could not compute fingerprints for initial population")
        else:
            print(f"Error: Could not load initial population from {args.initial_molecules}")

def process_data(input_dir, output_dir, random_state, plot_results=True, label="", 
                prefix_to_display=None, display_to_prefix=None, logp_range=None, umap_params=None, num_samples=1, initial_molecules_file=None):
    """Process data for a specific input directory and optionally create plots.
    
    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to save output plots
        random_state: Random seed for reproducibility
        plot_results: Whether to generate plots
        label: Optional label for logging
        prefix_to_display: Optional dictionary mapping from file prefix to display name
        display_to_prefix: Optional dictionary mapping from display name to file prefix
        logp_range: Optional list with [min, max] values for logP preferred range
        umap_params: Optional dictionary with UMAP parameters
        num_samples: Number of samples per method for error bars
        initial_molecules_file: Optional path to initial population file for CSV generation
        
    Returns:
        Tuple of (valid_data, fps_dict, mol_stats)
    """
    # Define mappings between file prefixes and internal keys
    file_prefix_to_key = {
        "alm_ppo": "ALM-RL",
        "alm": "ALM",
        "lm": "LM/LM-NG",     # Both LM and LM-NG map to same key
        "lm_ppo": "LM-RL",
        "lm_ng": "LM/LM-NG",  # Both LM and LM-NG map to same key
        "lm_ng_ppo": "LM-NG-RL"
    }
    
    # If custom display names are provided, override the internal keys
    # This will ensure we only use the models explicitly defined in the mapping file
    if prefix_to_display:
        # Clear the default mappings and only use those from the mapping file
        file_prefix_to_key = {}
        for prefix, display in prefix_to_display.items():
            file_prefix_to_key[prefix] = display
            print(f"Using custom display name: {prefix} → {display}")
    
    # Define file paths for processing multiple samples
    # For each type (valid_unique, valid, any), create a list of sample file paths
    
    # Valid unique population files (with validation and uniqueness)
    valid_unique_file_paths = {}
    for prefix, key in file_prefix_to_key.items():
        if key not in valid_unique_file_paths:
            valid_unique_file_paths[key] = []
        for sample_idx in range(1, num_samples + 1):
            sample_file = f"{input_dir}/{prefix}_valid_unique_only_{sample_idx}.csv"
            # Check if sample file exists
            if os.path.exists(sample_file):
                valid_unique_file_paths[key].append(sample_file)
            else:
                # Try the non-sample-indexed file as fallback
                fallback_file = f"{input_dir}/{prefix}_valid_unique_only.csv"
                if os.path.exists(fallback_file) and num_samples == 1:
                    valid_unique_file_paths[key].append(fallback_file)
                    print(f"Using fallback file for {key}: {fallback_file}")
    
    # Valid population files (with validation, duplicates allowed)
    valid_file_paths = {}
    for prefix, key in file_prefix_to_key.items():
        if key not in valid_file_paths:
            valid_file_paths[key] = []
        for sample_idx in range(1, num_samples + 1):
            sample_file = f"{input_dir}/{prefix}_valid_only_{sample_idx}.csv"
            if os.path.exists(sample_file):
                valid_file_paths[key].append(sample_file)
            else:
                fallback_file = f"{input_dir}/{prefix}_valid_only.csv"
                if os.path.exists(fallback_file) and num_samples == 1:
                    valid_file_paths[key].append(fallback_file)
                    print(f"Using fallback file for {key}: {fallback_file}")
    
    # Any population files (no validation)
    any_file_paths = {}
    for prefix, key in file_prefix_to_key.items():
        if key not in any_file_paths:
            any_file_paths[key] = []
        for sample_idx in range(1, num_samples + 1):
            sample_file = f"{input_dir}/{prefix}_any_{sample_idx}.csv"
            if os.path.exists(sample_file):
                any_file_paths[key].append(sample_file)
            else:
                fallback_file = f"{input_dir}/{prefix}_any.csv"
                if os.path.exists(fallback_file) and num_samples == 1:
                    any_file_paths[key].append(fallback_file)
                    print(f"Using fallback file for {key}: {fallback_file}")
    
    # Use valid_unique files for property analysis
    population_file_paths = valid_unique_file_paths
    
    # Use valid files for uniqueness calculation
    uniqueness_file_paths = valid_file_paths
    
    # Use any files for validity calculation
    validity_file_paths = any_file_paths

    # Calculate molecule statistics (validity and uniqueness) with error bars
    mol_stats = {}
    
    # Calculate validity from 'any' files
    for name, paths in validity_file_paths.items():
        if paths:  # Make sure we have at least one file
            sample_stats = []
            for path in paths:
                stats = calculate_mol_stats(path)
                if stats['validity'] > 0:
                    sample_stats.append(stats['validity'])
            
            if sample_stats:
                mean_validity = np.mean(sample_stats)
                std_validity = np.std(sample_stats) if len(sample_stats) > 1 else 0
                
                if name not in mol_stats:
                    mol_stats[name] = {}
                    
                mol_stats[name]['validity'] = mean_validity
                mol_stats[name]['validity_std'] = std_validity
                mol_stats[name]['validity_samples'] = sample_stats
                
                print(f"{name} validity: {mean_validity:.2f}% ± {std_validity:.2f}% (across {len(sample_stats)} samples)")
    
    # Calculate uniqueness from 'valid' files
    for name, paths in uniqueness_file_paths.items():
        if paths:
            sample_stats = []
            for path in paths:
                stats = calculate_mol_stats(path)
                if stats['uniqueness'] > 0:
                    sample_stats.append(stats['uniqueness'])
            
            if sample_stats:
                mean_uniqueness = np.mean(sample_stats)
                std_uniqueness = np.std(sample_stats) if len(sample_stats) > 1 else 0
                
                if name not in mol_stats:
                    mol_stats[name] = {}
                    
                mol_stats[name]['uniqueness'] = mean_uniqueness
                mol_stats[name]['uniqueness_std'] = std_uniqueness
                mol_stats[name]['uniqueness_samples'] = sample_stats
                
                print(f"{name} uniqueness: {mean_uniqueness:.2f}% ± {std_uniqueness:.2f}% (across {len(sample_stats)} samples)")
    
    # Load data for property analysis - Use all samples and calculate average property values
    loaded_data = {}
    # Dictionary to store UMAP data (will only use the first sample for UMAP)
    umap_data = {}
    
    for name, sample_paths in population_file_paths.items():
        if not sample_paths:
            continue
            
        # Process all sample data for properties (for averaging)
        all_method_dfs = []
        for path in sample_paths:
            df = load_data(path)
            if df is not None:
                all_method_dfs.append(df)
        
        if all_method_dfs:
            # Combine all samples into one DataFrame for averaging
            # Concatenate all samples
            combined_df = pd.concat(all_method_dfs, ignore_index=True)
            
            # Calculate average property values across all molecules
            avg_props = {}
            for prop in ['logP', 'drug', 'synth']:
                if prop in combined_df.columns:
                    avg_props[prop] = combined_df[prop].mean()
                    std_props = combined_df[prop].std()
                    print(f"{name} average {prop}: {avg_props[prop]:.3f} ± {std_props:.3f}")
            
            # Store the combined data for property analysis
            loaded_data[name] = combined_df
            print(f"Combined {len(all_method_dfs)} samples for {name} with total {len(combined_df)} molecules")
            
            # For UMAP visualization, only use the first sample
            first_sample_path = None
            for path in sample_paths:
                if '_1.csv' in path:
                    first_sample_path = path
                    break
            
            # If no _1.csv found, use the first available sample
            if first_sample_path is None and sample_paths:
                first_sample_path = sample_paths[0]
            
            if first_sample_path:
                umap_df = load_data(first_sample_path)
                if umap_df is not None:
                    umap_data[name] = umap_df
                    print(f"Using {first_sample_path} for UMAP visualization ({len(umap_df)} molecules)")
    
    # Filter out None entries if files weren't loaded successfully
    valid_data = {name: df for name, df in loaded_data.items() if df is not None}
    valid_umap_data = {name: df for name, df in umap_data.items() if df is not None}

    if not valid_data:
        print(f"Error: No valid data loaded from {input_dir}. Exiting.")
        return None, None, None

    # Define properties to plot
    properties_to_plot = ['synth', 'drug', 'logP']
    
    # Calculate fitness scores
    fitness_stats = calculate_fitness_scores_clean(valid_data)
    
    # Create plots if requested
    if plot_results:
        # Plot distributions
        plot_distributions(valid_data, properties_to_plot, mol_stats=mol_stats, output_dir=output_dir, logp_lines=logp_range)
        
        # Plot molecular properties only (SA, QED, logP) without validity/uniqueness
        plot_molecular_properties_only(valid_data, properties_to_plot, output_dir=output_dir, logp_lines=logp_range)
        
        # Create UMAP visualizations using only first sample data
        fps_dict = compute_fingerprints_for_dataframes(valid_umap_data)
        plot_umap_visualization(fps_dict, properties_to_plot, output_dir=output_dir, random_state=random_state, umap_params=umap_params)
        
        # Create combined methods UMAP visualization
        plot_combined_methods_umap(fps_dict, output_dir=output_dir, random_state=random_state, umap_params=umap_params)
        
        # Save performance tables (validity, uniqueness, fitness)
        save_performance_tables_clean(mol_stats, fitness_stats, output_dir)
        
        # Save molecules with properties and fitness scores to CSV files
        save_molecules_with_properties_and_fitness(valid_data, output_dir)
        
        # Generate combined paper figure
        plot_paper_figure(valid_data, fps_dict, mol_stats, output_dir=output_dir, 
                         logp_lines=logp_range, random_state=random_state, umap_params=umap_params)
    
    # Process initial population for CSV generation if file is provided
    if initial_molecules_file and os.path.exists(initial_molecules_file):
        print(f"\nProcessing initial population from {initial_molecules_file} for CSV generation...")
        initial_df = load_data(initial_molecules_file)
        if initial_df is not None:
            initial_data = {"Initial": initial_df}
            print("Generating CSV with properties and fitness for initial population...")
            save_molecules_with_properties_and_fitness(initial_data, output_dir)
        else:
            print(f"Error: Could not load initial population from {initial_molecules_file}")
    
    return valid_data, fps_dict, mol_stats

def calculate_fitness_scores_clean(data_dict):
    """
    Calculate fitness scores for each method using harmonic mean of properties.
    
    Args:
        data_dict: Dictionary of DataFrames with molecular property data
        
    Returns:
        Dictionary with fitness statistics for each method
    """
    fitness_stats = {}
    
    for method_name, df in data_dict.items():
        if df is None or df.empty:
            continue
            
        # Skip initial population
        if method_name.lower() == 'initial':
            continue
            
        # Properties for fitness calculation
        required_props = ['synth', 'drug', 'logP']
        
        # Check if all required properties are available
        if not all(prop in df.columns for prop in required_props):
            print(f"Warning: Missing properties for fitness calculation in {method_name}")
            continue
        
        # Calculate fitness for each molecule
        fitness_scores = []
        
        for _, row in df.iterrows():
            # Get property values
            sa_score = row['synth']  # Synthetic accessibility (1-7, lower is better)
            qed_score = row['drug']  # Drug-likeness (0-1, higher is better)
            logp_value = row['logP']  # Lipophilicity
            
            # Skip if any property is NaN
            if pd.isna(sa_score) or pd.isna(qed_score) or pd.isna(logp_value):
                continue
            
            # Normalize properties to 0-1 scale (higher is better)
            # SA Score: Transform so that lower SA (easier synthesis) gives higher score
            sa_normalized = max(0, (7 - sa_score) / 6)  # Transform 1-7 to 1-0, then to 0-1
            
            # QED is already 0-1, higher is better
            qed_normalized = max(0, min(1, qed_score))
            
            # LogP: Optimal range is typically 1-3 for drug-like molecules
            # Use a penalty function that gives maximum score at logP = 2
            if logp_value <= 2:
                logp_normalized = max(0, logp_value / 2)  # Linear increase from 0 to 1
            else:
                logp_normalized = max(0, 1 - (logp_value - 2) / 4)  # Linear decrease, 0 at logP = 6
            
            logp_normalized = max(0, min(1, logp_normalized))
            
            # Calculate harmonic mean of normalized properties
            properties = [sa_normalized, qed_normalized, logp_normalized]
            if all(p > 0 for p in properties):
                fitness = len(properties) / sum(1/p for p in properties)
            else:
                fitness = 0  # If any property is 0, fitness is 0
            
            fitness_scores.append(fitness)
        
        # Calculate statistics
        if fitness_scores:
            fitness_mean = np.mean(fitness_scores)
            fitness_std = np.std(fitness_scores) if len(fitness_scores) > 1 else 0
            
            fitness_stats[method_name] = {
                'fitness_mean': fitness_mean,
                'fitness_std': fitness_std,
                'fitness_scores': fitness_scores,
                'num_molecules': len(fitness_scores)
            }
            
            print(f"{method_name} fitness: {fitness_mean:.3f} ± {fitness_std:.3f} (n={len(fitness_scores)})")
        else:
            print(f"No valid fitness scores calculated for {method_name}")
    
    return fitness_stats

def save_performance_tables_clean(mol_stats, fitness_stats, output_dir):
    """
    Save performance metrics (validity, uniqueness, fitness) to CSV and formatted text tables.
    
    Args:
        mol_stats: Dictionary with validity and uniqueness statistics
        fitness_stats: Dictionary with fitness statistics
        output_dir: Directory to save the tables
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all method names from both dictionaries
    all_methods = set()
    if mol_stats:
        all_methods.update(mol_stats.keys())
    if fitness_stats:
        all_methods.update(fitness_stats.keys())
    
    # Remove 'initial' or 'Initial' from method ordering for main table
    methods_to_include = [m for m in all_methods if m.lower() != 'initial']
    
    # Create a comprehensive performance table
    performance_data = []
    
    for method in sorted(methods_to_include):
        row = {'Method': method}
        
        # Add validity metrics
        if mol_stats and method in mol_stats and 'validity' in mol_stats[method]:
            validity_mean = mol_stats[method]['validity']
            validity_std = mol_stats[method].get('validity_std', 0)
            row['Validity (%)'] = f"{validity_mean:.1f}"
            if validity_std > 0:
                row['Validity_Std'] = f"{validity_std:.1f}"
        else:
            row['Validity (%)'] = "N/A"
            row['Validity_Std'] = "N/A"
        
        # Add uniqueness metrics
        if mol_stats and method in mol_stats and 'uniqueness' in mol_stats[method]:
            uniqueness_mean = mol_stats[method]['uniqueness']
            uniqueness_std = mol_stats[method].get('uniqueness_std', 0)
            row['Uniqueness (%)'] = f"{uniqueness_mean:.1f}"
            if uniqueness_std > 0:
                row['Uniqueness_Std'] = f"{uniqueness_std:.1f}"
        else:
            row['Uniqueness (%)'] = "N/A"
            row['Uniqueness_Std'] = "N/A"
        
        # Add fitness metrics
        if fitness_stats and method in fitness_stats:
            fitness_mean = fitness_stats[method].get('fitness_mean', 0)
            fitness_std = fitness_stats[method].get('fitness_std', 0)
            row['Fitness Score'] = f"{fitness_mean:.3f}"
            if fitness_std > 0:
                row['Fitness_Std'] = f"{fitness_std:.3f}"
        else:
            row['Fitness Score'] = "N/A"
            row['Fitness_Std'] = "N/A"
        
        performance_data.append(row)
    
    # Create DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "performance_metrics.csv")
    performance_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to: {csv_path}")
    
    # Create a formatted text table for easy reading
    txt_path = os.path.join(output_dir, "performance_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("Performance Metrics Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Create a nicely formatted table
        f.write(f"{'Method':<12} {'Validity (%)':<12} {'Uniqueness (%)':<15} {'Fitness Score':<15}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in performance_df.iterrows():
            method = row['Method']
            validity = row['Validity (%)']
            uniqueness = row['Uniqueness (%)']
            fitness = row['Fitness Score']
            
            # Add error bars if available
            validity_std = row.get('Validity_Std', 'N/A')
            uniqueness_std = row.get('Uniqueness_Std', 'N/A')
            fitness_std = row.get('Fitness_Std', 'N/A')
            
            validity_display = format_metric_with_std(validity, validity_std)
            uniqueness_display = format_metric_with_std(uniqueness, uniqueness_std)
            fitness_display = format_metric_with_std(fitness, fitness_std)
            
            f.write(f"{method:<12} {validity_display:<12} {uniqueness_display:<15} {fitness_display:<15}\n")
        
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Validity: Percentage of generated molecules that are chemically valid\n")
        f.write("- Uniqueness: Percentage of valid molecules that are unique (non-duplicated)\n")
        f.write("- Fitness Score: Harmonic mean of normalized molecular properties\n")
        f.write("- Error bars (±) shown when multiple samples are available\n")
    
    print(f"Formatted performance table saved to: {txt_path}")
    
    # Create LaTeX table format
    latex_path = os.path.join(output_dir, "performance_metrics_latex.txt")
    with open(latex_path, 'w') as f:
        f.write("% LaTeX table for performance metrics\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\caption{Performance metrics for molecular generation}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Validity (\\%) & Uniqueness (\\%) & Fitness Score \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in performance_df.iterrows():
            method = row['Method']
            validity = row['Validity (%)']
            uniqueness = row['Uniqueness (%)']
            fitness = row['Fitness Score']
            
            f.write(f"{method} & {validity} & {uniqueness} & {fitness} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_path}")

def plot_paper_figure(data_dict, fps_dict, mol_stats, output_dir="plots", logp_lines=None, random_state=42, umap_params=None):
    """
    Create a combined figure for paper with validity, UMAP, and molecular properties.
    Layout: 2 rows x 3 columns
    Row 1: [Validity bar chart] [UMAP] [Empty]
    Row 2: [SA distribution] [QED distribution] [logP distribution]
    
    Args:
        data_dict: Dictionary of DataFrames with molecular property data
        fps_dict: Dictionary with fingerprints for each method
        mol_stats: Dictionary with validity and uniqueness statistics
        output_dir: Directory to save the plot
        logp_lines: Optional list with logP preferred range
        random_state: Random state for UMAP
        umap_params: Optional dictionary with UMAP parameters
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating combined paper figure...")
    
    # Set up the figure with a custom layout: top row centered, bottom row full width
    fig = plt.figure(figsize=(15, 10))
    
    # Create custom subplot layout with reduced spacing
    # Top row: 2 centered plots with some spacing
    ax_validity = plt.subplot2grid((2, 6), (0, 1), colspan=2)  # Centered validity plot
    ax_umap = plt.subplot2grid((2, 6), (0, 3), colspan=2)      # Centered UMAP plot
    
    # Bottom row: 3 full-width property plots
    ax_sa = plt.subplot2grid((2, 6), (1, 0), colspan=2)        # SA distribution
    ax_qed = plt.subplot2grid((2, 6), (1, 2), colspan=2)       # QED distribution  
    ax_logp = plt.subplot2grid((2, 6), (1, 4), colspan=2)      # logP distribution
    
    # Store axes for easy access
    property_axes = [ax_sa, ax_qed, ax_logp]
    
    # Set font sizes for paper figure
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 14  # Increased from 10 to 14
    })
    
    # Get centralized color mapping
    all_method_colors = get_method_color_mapping()
    
    # Get available methods (excluding initial) in preferred order
    all_available_methods = [m for m in data_dict.keys() if m.lower() != 'initial']
    expected_order = ["LM/LM-NG", "LM-NG-RL", "LM-RL", "ALM", "ALM-RL", "Fixed-LM", "Fixed-LM-NG"]
    
    # Order methods according to expected order
    available_methods = []
    for method in expected_order:
        if method in all_available_methods:
            available_methods.append(method)
    
    # Add any remaining methods not in expected order
    for method in all_available_methods:
        if method not in available_methods:
            available_methods.append(method)
    
    # ========== Top Left: Validity Bar Chart ==========
    if mol_stats:
        # Get validity data
        names = []
        values = []
        errors = []
        bar_colors = []
        
        for name in available_methods:
            # Get the key to use for mol_stats lookup
            stats_key = name
            if name not in mol_stats:
                # Try finding a case-insensitive match
                matching_keys = [k for k in mol_stats.keys() if k.lower() == name.lower()]
                if matching_keys:
                    stats_key = matching_keys[0]
                else:
                    continue
            
            if 'validity' in mol_stats[stats_key]:
                names.append(name)
                values.append(mol_stats[stats_key]['validity'] / 100.0)  # Convert to 0-1 scale
                
                # Get error value if available
                if 'validity_std' in mol_stats[stats_key]:
                    errors.append(mol_stats[stats_key]['validity_std'] / 100.0)
                else:
                    errors.append(0)
                    
                bar_colors.append(all_method_colors.get(name, "#808080"))
        
        # Create bar chart
        bars = ax_validity.bar(names, values, yerr=errors, color=bar_colors, 
                              edgecolor='black', linewidth=1, capsize=5, error_kw={'elinewidth': 1.5})
        
        ax_validity.set_ylabel('Validity', fontweight='bold')
        ax_validity.set_ylim(0, 1)  # Set y-axis from 0 to 1 for validity
        ax_validity.grid(axis='y', linestyle='--', alpha=0.7)
        # No title - removed as requested
        
        # Rotate x-axis labels if needed
        ax_validity.tick_params(axis='x', rotation=45)
    
    # Add subplot label for validity plot
    ax_validity.text(0.02, 0.98, 'a', transform=ax_validity.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    
    # Make validity plot square
    # ax_validity.set_aspect('equal', adjustable='box')  # Commented out - interferes with natural scaling
    
    # ========== Top Right: UMAP ==========
    if fps_dict:
        # Get methods available in fps_dict (excluding initial) in preferred order
        all_umap_methods = [m for m in fps_dict.keys() if m.lower() != 'initial']
        umap_methods = []
        for method in expected_order:
            if method in all_umap_methods:
                umap_methods.append(method)
        # Add any remaining methods not in expected order
        for method in all_umap_methods:
            if method not in umap_methods:
                umap_methods.append(method)
        
        if umap_methods:
            # Define different marker shapes for different methods
            markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'P', 'X']
            method_markers = {method: markers[i % len(markers)] for i, method in enumerate(umap_methods)}
            
            # Combine all fingerprints
            all_fps = []
            method_labels = []
            
            for method in umap_methods:
                if method in fps_dict:
                    method_data = fps_dict[method]
                    method_fps = method_data['fps']
                    
                    all_fps.append(method_fps)
                    method_labels.extend([method] * len(method_fps))
            
            if all_fps:
                # Concatenate all fingerprints
                combined_fps = np.vstack(all_fps)
                
                # Use provided UMAP parameters or defaults
                default_umap_params = {
                    "n_neighbors": 5,
                    "min_dist": 0.1,
                    "metric": "jaccard"
                }
                
                umap_config = default_umap_params.copy()
                if umap_params:
                    umap_config.update(umap_params)
                
                # Apply UMAP
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=umap_config["n_neighbors"],
                    min_dist=umap_config["min_dist"],
                    metric=umap_config["metric"],
                    random_state=random_state
                )
                embedding = reducer.fit_transform(combined_fps)
                
                # Plot each method with consistent colors and different markers
                for method in umap_methods:
                    indices = [i for i, label in enumerate(method_labels) if label == method]
                    if indices:
                        color = all_method_colors.get(method, "#808080")
                        marker = method_markers[method]
                        ax_umap.scatter(
                            embedding[indices, 0], 
                            embedding[indices, 1],
                            c=color,
                            marker=marker,
                            label=method,
                            s=60,  # Slightly larger for better visibility
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5
                        )
                
                ax_umap.set_xlabel("UMAP 1")
                ax_umap.set_ylabel("UMAP 2")
                # No title - removed as requested
                
                # Set integer tick labels for UMAP axes
                ax_umap.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax_umap.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                # Position legend outside the plot area to avoid overlap with larger font
                ax_umap.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))
                ax_umap.grid(linestyle='--', alpha=0.3)
    
    # Add subplot label for UMAP plot
    ax_umap.text(0.02, 0.98, 'b', transform=ax_umap.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    
    # Make UMAP plot square
    # ax_umap.set_aspect('equal', adjustable='box')  # Commented out - interferes with natural scaling
    
    # ========== Bottom Row: Molecular Properties ==========
    properties = ['synth', 'drug', 'logP']
    property_labels = {
        'logP': 'logP',
        'drug': 'QED',
        'synth': 'SA'
    }
    
    for i, prop in enumerate(properties):
        ax = property_axes[i]
        plot_label = property_labels.get(prop, prop)
        
        # Plot property distributions
        plotted_something = False
        for method_name in available_methods:
            if method_name in data_dict and data_dict[method_name] is not None:
                if prop in data_dict[method_name].columns:
                    valid_data = data_dict[method_name][prop].dropna()
                    if not valid_data.empty:
                        color = all_method_colors.get(method_name, "#808080")
                        sns.kdeplot(valid_data, ax=ax, label=method_name, color=color, 
                                   fill=False, linewidth=2)
                        plotted_something = True
        
        if plotted_something:
            ax.set_xlabel(plot_label, fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.grid(linestyle='--', alpha=0.7)
            
            # Add reference lines and set axis limits
            if prop == 'logP':
                # Add vertical lines for logP preferred range
                if logp_lines and isinstance(logp_lines, list):
                    for logp_value in logp_lines:
                        ax.axvline(x=logp_value, color='black', linestyle='--', linewidth=1)
                else:
                    ax.axvline(x=2, color='black', linestyle='--', linewidth=1)
                    ax.axvline(x=3, color='black', linestyle='--', linewidth=1)
                # Auto-scale both axes for logP
                ax.autoscale(enable=True, axis='both', tight=False)
            elif prop == 'drug':
                # Auto-scale y-axis only, then set x-axis limits
                ax.autoscale(enable=True, axis='y', tight=False)
                ax.set_xlim(0, 1)  # QED range from 0 to 1
            elif prop == 'synth':
                # Auto-scale y-axis only, then set x-axis limits
                ax.autoscale(enable=True, axis='y', tight=False)
                ax.set_xlim(1, 7)  # SA Score range from 1 to 7
            
            # Add legend only to the first property plot with larger font
            if i == 0:
                ax.legend(fontsize=12, loc='upper right')

    # Add subplot labels for property plots (c, d, e)
    subplot_labels = ['c', 'd', 'e']
    for i, ax in enumerate(property_axes):
        ax.text(0.02, 0.98, subplot_labels[i], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

    # Make all property plots square
    # for ax in property_axes:
    #     ax.set_aspect('equal', adjustable='box')  # Commented out - interferes with natural scaling

    # Adjust layout with better spacing between subplots
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, 
                       wspace=0.4, hspace=0.4)  # Increased spacing for better readability
    
    # Force all plots to be square by setting box aspect
    ax_validity.set_box_aspect(1)
    ax_umap.set_box_aspect(1)
    for ax in property_axes:
        ax.set_box_aspect(1)
    
    # Clear any axis limits to allow natural scaling
    ax_validity.relim()
    ax_validity.autoscale_view()
    ax_umap.relim() 
    ax_umap.autoscale_view()
    # For property axes, only autoscale y-axis to preserve x-limits set above
    for ax in property_axes:
        ax.relim()
        ax.autoscale_view(scaley=True, scalex=False)  # Only scale y-axis
    
    plt.savefig(os.path.join(output_dir, "paper_figure.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Paper figure saved to: {os.path.join(output_dir, 'paper_figure.png')}")
    
    # Reset matplotlib defaults
    plt.rcdefaults()

if __name__ == "__main__":
    main() 