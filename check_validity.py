#!/usr/bin/env python
import pandas as pd
import os
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def check_molecule_validity(filepath):
    """
    Checks the validity of molecules in a CSV file containing SMILES strings.
    
    Args:
        filepath: Path to CSV/TSV file containing 'smiles' column
    
    Returns:
        Dictionary with validity statistics
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    
    try:
        # Determine separator based on file extension
        if filepath.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t', engine='python')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=',')
        else:
            print(f"Error: Unsupported file format for {filepath}")
            return None
        
        if 'smiles' not in df.columns:
            print(f"Error: 'smiles' column not found in {filepath}")
            return None
        
        print(f"Checking validity of molecules in {filepath}...")
        
        # Check validity and track unique valid SMILES
        valid_count = 0
        invalid_count = 0
        total_count = len(df)
        unique_smiles = set()
        invalid_smiles = []
        
        for smi in tqdm(df['smiles'], desc="Checking molecules"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_count += 1
                # Get canonical SMILES to ensure consistency in uniqueness check
                canonical_smi = Chem.MolToSmiles(mol)
                unique_smiles.add(canonical_smi)
            else:
                invalid_count += 1
                invalid_smiles.append(smi)
        
        # Calculate percentages
        valid_percent = (valid_count / total_count) * 100 if total_count > 0 else 0
        unique_percent = (len(unique_smiles) / valid_count) * 100 if valid_count > 0 else 0
        duplicate_count = valid_count - len(unique_smiles)
        duplicate_percent = (duplicate_count / valid_count) * 100 if valid_count > 0 else 0
        
        # Print results
        print("\nValidity Statistics:")
        print(f"Total molecules: {total_count}")
        print(f"Valid molecules: {valid_count} ({valid_percent:.2f}%)")
        print(f"Invalid molecules: {invalid_count} ({100-valid_percent:.2f}%)")
        print(f"Unique valid molecules: {len(unique_smiles)} ({unique_percent:.2f}% of valid)")
        print(f"Duplicate molecules: {duplicate_count} ({duplicate_percent:.2f}% of valid)")
        
        # If there are invalid SMILES, print the first few
        if invalid_smiles:
            print("\nSample of invalid SMILES (first 5):")
            for i, smi in enumerate(invalid_smiles[:5]):
                print(f"{i+1}. {smi}")
        
        return {
            'total_count': total_count,
            'valid_count': valid_count,
            'valid_percent': valid_percent,
            'invalid_count': invalid_count,
            'unique_count': len(unique_smiles),
            'unique_percent': unique_percent,
            'duplicate_count': duplicate_count,
            'duplicate_percent': duplicate_percent,
            'invalid_smiles': invalid_smiles
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

if __name__ == "__main__":
    # File to check
    file_path = "output_inference/alm_valid.csv"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found. Please make sure the file exists.")
    else:
        # Check validity
        stats = check_molecule_validity(file_path)
        
        # If you want to save the invalid SMILES to a file for further analysis
        if stats and stats['invalid_smiles']:
            invalid_file = "invalid_molecules.txt"
            print(f"\nSaving {len(stats['invalid_smiles'])} invalid SMILES to {invalid_file}")
            with open(invalid_file, 'w') as f:
                for smi in stats['invalid_smiles']:
                    f.write(f"{smi}\n") 