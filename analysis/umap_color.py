import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer
import umap
import time
from admet_ai import ADMETModel
from matplotlib.gridspec import GridSpec

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

def compute_fingerprints(smiles_list):
    """
    Compute fingerprints for a list of SMILES strings.
    
    Args:
        smiles_list (list): List of SMILES strings.
        
    Returns:
        tuple: (fingerprints, valid_indices) where fingerprints is a list of numpy arrays 
               and valid_indices are the indices of valid SMILES in the original list.
    """
    fps = []
    valid_indices = []
    for i, s in enumerate(smiles_list):
        fp = smiles_to_fingerprint(s)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    return fps, valid_indices

def compute_properties(smiles_list):
    """
    Compute synthesizability, QED, logP, and atom count for a list of SMILES strings.
    
    Returns:
        tuple: (synth_scores, qed_scores, logp_values, atom_counts, valid_indices)
    """
    synth_scores = []
    qed_scores = []
    logp_values = []
    atom_counts = []
    valid_indices = []
    
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        synth_scores.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))
        logp_values.append(Descriptors.MolLogP(mol))
        atom_counts.append(mol.GetNumHeavyAtoms())
        valid_indices.append(i)
        
    return synth_scores, qed_scores, logp_values, atom_counts, valid_indices

def predict_permeability(smiles_list, model, admet_key, valid_indices):
    """
    Predict permeability for a list of SMILES strings using a given ADMET model.
    
    Args:
        smiles_list (list): List of SMILES strings.
        model (ADMETModel): An instance of ADMETModel.
        admet_key (str): The key to extract ADMET predictions.
        valid_indices (list): List of indices of valid molecules.
    
    Returns:
        list: Permeability values corresponding to the admet_key.
    """
    # Only compute predictions for valid molecules
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    start_time = time.time()
    predictions = model.predict(smiles=valid_smiles)
    end_time = time.time()
    print(f"Permeability prediction took {end_time - start_time:.2f} seconds.")
    
    return predictions[admet_key].tolist()

def plot_property_umap(embedding, property_values, labels, title, property_name, cmap='viridis', invert=False, ax=None, vmin=None, vmax=None):
    """
    Plot UMAP embedding with property-based coloring.
    
    Args:
        embedding (np.array): 2D UMAP embedding.
        property_values (list): Property values to color by.
        labels (list): Labels for points (Original/Generated).
        title (str): Plot title.
        property_name (str): Name of the property for labeling.
        cmap (str): Matplotlib colormap to use.
        invert (bool): Whether to invert the colormap.
        ax (matplotlib.axes): Axes to plot on.
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Create scatter plot for original and generated molecules
    # (Different markers, but color based on property)
    orig_mask = np.array([l == "Original" for l in labels])
    gen_mask = np.array([l == "Generated" for l in labels])
    
    # Normalize color values - but keep track of actual values for colorbar
    property_array = np.array(property_values)
    if invert:
        cmap = f"{cmap}_r"  # Use reversed colormap if needed
    
    # Original molecules
    orig_scatter = ax.scatter(
        embedding[orig_mask, 0], embedding[orig_mask, 1],
        c=property_array[orig_mask], cmap=cmap, vmin=vmin, vmax=vmax,
        marker='o', s=25, alpha=0.7, 
    )
    
    # Generated molecules
    gen_scatter = ax.scatter(
        embedding[gen_mask, 0], embedding[gen_mask, 1],
        c=property_array[gen_mask], cmap=cmap, vmin=vmin, vmax=vmax,
        marker='*', s=25, alpha=0.7
    )
    
    # Colorbar with actual property values (not normalized)
    cbar = plt.colorbar(orig_scatter, ax=ax)
    cbar.set_label(property_name, size=14)
    
    # Title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    
    # Create a custom legend for molecule types (Original/Generated)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Original'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=14, label='Generated')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    return ax

def main():
    parser = argparse.ArgumentParser(
        description="UMAP projection with property-based coloring."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="../data/linker_smiles.csv",
        help="Path to CSV file with original linker SMILES"
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default="../output/linker_fine_tuning_admet/linker_admet_run_population.tsv",
        help="Path to TSV file with generated linker SMILES"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./umap_property_plots",
        help="Directory for output figures"
    )
    parser.add_argument(
        "--admet_key",
        type=str,
        default="Caco2_Wang",
        help="The key to use for ADMET prediction from the ADMET model output"
    )
    parser.add_argument(
        "--combined_figure",
        action="store_true",
        help="Create a combined figure with all properties"
    )
    
    args = parser.parse_args()
    
    # --- Process original linkers ---
    df_orig = pd.read_csv(args.csv_file, sep="\s+")
    orig_smiles = df_orig['smiles'].tolist()
    
    # --- Process generated molecules ---
    df_gen = pd.read_csv(args.tsv_file, sep="\t")
    gen_smiles = df_gen['smiles'].tolist()
    
    # Combine all SMILES
    all_smiles = orig_smiles + gen_smiles
    print(f"Processing {len(all_smiles)} molecules total ({len(orig_smiles)} original, {len(gen_smiles)} generated)")
    
    # Compute fingerprints for UMAP
    all_fps, valid_indices = compute_fingerprints(all_smiles)
    print(f"Computed fingerprints for {len(all_fps)} valid molecules")
    
    # Compute properties for valid molecules
    valid_smiles = [all_smiles[i] for i in valid_indices]
    print(f"Computing properties for {len(valid_smiles)} valid molecules")
    synth_scores, qed_scores, logp_values, atom_counts, prop_valid_indices = compute_properties(valid_smiles)
    
    # Ensure all indices match
    if len(prop_valid_indices) != len(valid_indices):
        print("Warning: Some molecules could be processed for fingerprints but not properties")
        # In a real-world scenario, you'd handle this more carefully
    
    # Compute permeability for valid molecules
    model = ADMETModel()
    permeability_values = predict_permeability(valid_smiles, model, args.admet_key, prop_valid_indices)
    
    # Create labels for visualization
    labels = []
    for idx in valid_indices:
        if idx < len(orig_smiles):
            labels.append("Original")
        else:
            labels.append("Generated")
    
    # Apply UMAP to fingerprints
    print("Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.3,
        metric="jaccard",
        random_state=42
    )
    embedding = reducer.fit_transform(all_fps)
    print("UMAP embedding complete")
    
    # Create output directory if it doesn't exist
    import os
    output_dir = "./umap_property_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define property configurations for plotting with fixed ranges
    property_configs = [
        {
            "values": synth_scores,
            "name": "SA Score",
            "title": "UMAP - Synthetic Accessibility Score",
            "cmap": "viridis",
            "invert": False,  # Invert so darker = lower SA (more synthesizable)
            "filename": f"{output_dir}/umap_sa_score.png",
            "vmin": 1,
            "vmax": 4
        },
        {
            "values": qed_scores,
            "name": "QED Score",
            "title": "UMAP - Drug-likeness (QED)",
            "cmap": "viridis",
            "invert": False,  # Darker = higher QED
            "filename": f"{output_dir}/umap_qed.png",
            "vmin": 0,
            "vmax": 1
        },
        {
            "values": logp_values,
            "name": "logP",
            "title": "UMAP - logP",
            "cmap": "viridis",
            "invert": False,  # Darker = higher logP
            "filename": f"{output_dir}/umap_logp.png",
            "vmin": -2,
            "vmax": 6
        },
        {
            "values": permeability_values,
            "name": "Permeability",
            "title": "UMAP - Permeability",
            "cmap": "viridis",
            "invert": False,  # Darker = higher permeability
            "filename": f"{output_dir}/umap_permeability.png",
            "vmin": -6,
            "vmax": -3
        },
        {
            "values": atom_counts,
            "name": "Heavy Atom Count",
            "title": "UMAP - Heavy Atom Count",
            "cmap": "viridis",
            "invert": False,  # Darker = higher atom count
            "filename": f"{output_dir}/umap_atom_count.png",
            "vmin": 0,
            "vmax": 40
        }
    ]
    
    if args.combined_figure:
        # Create a combined figure with all properties
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        for i, config in enumerate(property_configs):
            if i < 5:  # We have 5 properties, so map to the 2x3 grid
                row, col = divmod(i, 3)
                ax = fig.add_subplot(gs[row, col])
                plot_property_umap(
                    embedding, config["values"], labels,
                    config["title"], config["name"],
                    config["cmap"], config["invert"], ax,
                    vmin=config["vmin"], vmax=config["vmax"]
                )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_all_properties.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Create individual figures for each property
        for config in property_configs:
            plt.figure(figsize=(10, 8))
            plot_property_umap(
                embedding, config["values"], labels,
                config["title"], config["name"],
                config["cmap"], config["invert"],
                vmin=config["vmin"], vmax=config["vmax"]
            )
            plt.tight_layout()
            plt.savefig(config["filename"], dpi=300, bbox_inches='tight')
            plt.close()
    
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()