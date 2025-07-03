import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import umap

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
        list: List of fingerprint vectors (np.array), excluding any that could not be computed.
    """
    fps = []
    for s in smiles_list:
        fp = smiles_to_fingerprint(s)
        if fp is not None:
            fps.append(fp)
    return fps

def main():
    parser = argparse.ArgumentParser(
        description="UMAP projection of molecular fingerprints from original and generated linkers."
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
        "--output_fig",
        type=str,
        default="./umap_plot.png",
        help="Filename for the output UMAP projection figure"
    )
    
    args = parser.parse_args()
    
    # --- Process original linkers ---
    df_orig = pd.read_csv(args.csv_file, sep="\s+")
    orig_smiles = df_orig['smiles'].tolist()
    orig_fps = compute_fingerprints(orig_smiles)
    print(f"Computed fingerprints for {len(orig_fps)} original molecules.")
    
    # --- Process generated molecules ---
    df_gen = pd.read_csv(args.tsv_file, sep="\t")
    gen_smiles = df_gen['smiles'].tolist()
    gen_fps = compute_fingerprints(gen_smiles)
    print(f"Computed fingerprints for {len(gen_fps)} generated molecules.")
    
    # Combine fingerprints and prepare labels.
    # We'll keep track of the index so we can split the embedding later.
    all_fps = np.vstack(orig_fps + gen_fps)
    labels = (["Original"] * len(orig_fps)) + (["Generated"] * len(gen_fps))
    
    # --- Apply UMAP ---
    reducer = umap.UMAP(
    n_components=2,
    n_neighbors=5,   # Try smaller or larger
    min_dist=0.3,     # Allow points to be very close
    metric="jaccard", # Good for binary fingerprints
    random_state=42
    )
    embedding = reducer.fit_transform(all_fps)

    # reducer = umap.UMAP(n_components=2, random_state=42)
    # embedding = reducer.fit_transform(all_fps)
    
    # --- Plot the UMAP projection ---
    # Separate the embeddings based on original vs generated.
    emb_orig = embedding[:len(orig_fps)]
    emb_gen = embedding[len(orig_fps):]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_orig[:, 0], emb_orig[:, 1], c='black', label='Original', alpha=0.5)
    plt.scatter(emb_gen[:, 0], emb_gen[:, 1], c='red', label='Generated', alpha=0.5)

    # Increase font size for title and axes labels
    plt.title("UMAP Projection of Molecular Fingerprints", fontsize=16)
    plt.xlabel("UMAP 1", fontsize=18)
    plt.ylabel("UMAP 2", fontsize=18)

    # Increase font size for tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(args.output_fig)
    plt.show()
    
if __name__ == '__main__':
    main()
