import torch
import argparse
import json
import os
import sys
import csv
import random
from rdkit import Chem
from rdkit import RDLogger
from models.gan import Gan
from utils.util import (
    setup_device_and_logging,
    initialize_gan_operators,
    initialize_scaffold_handler,
    parse_mutation_params
)
from utils.training_utils import setup_scoring_operator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Molecule inference with LM-GAN')

    # Input/Output
    parser.add_argument('--config', type=str, default='./config/default.json', 
                        help='json config for scoring operators and gans')
    parser.add_argument('--output_directory', type=str, default='./output_inference', 
                        help='output directory for generated molecules')
    parser.add_argument('--data_file', type=str, default='./output_inference/generated_molecules.csv', 
                        help='path to save generated SMILES')
    parser.add_argument('--initial_molecules_file', type=str, default=None,
                        help='file with initial SMILES data in first column for non-scaffold generation')
    parser.add_argument('--run_id', type=str, default='inference', 
                        help='run id used as a prefix in output files')
    parser.add_argument('--model_file', type=str, default=None, 
                        help='path to saved model .pt file')
    
    # Generation parameters
    parser.add_argument('--sample_size', type=int, default=1000, 
                        help='number of molecules to generate')
    parser.add_argument('--mutation_parameter', type=str, default='1.0', 
                        help='determines fraction of tokens that are masked for generator')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='batch size for generation')
    parser.add_argument('--top_k', type=int, default=5, 
                        help='number of top predictions for generator evaluation')
    
    # Model settings
    parser.add_argument('--generator_only', action='store_true', default=True, 
                        help='option to use only generator')
    parser.add_argument('--lr', type=float, default=0.00002, 
                        help='learning rate for model training')
    
    # Scaffold options
    parser.add_argument('--use_scaffold', action='store_true', default=False,
                       help='option to use scaffold-based generation')
    parser.add_argument('--mask_mode', type=str, default="sample_partition", 
                        choices=["replace", "random", "sample_partition"],
                        help='Mode for masking tokens: replace or sample_partition')
                        
    # Validation options - mutually exclusive
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument('--valid_unique_only', action='store_true', default=False,
                       help='option to validate that generated molecules contain scaffold and are unique')
    validation_group.add_argument('--valid_only', action='store_true', default=False,
                       help='option to validate that generated molecules contain scaffold (duplicates allowed)')
                        
    return parser.parse_args()

def load_initial_molecules(file_path, delimiter='\t'):
    """Load initial molecules from a file.
    
    Args:
        file_path: Path to the file with SMILES strings
        delimiter: Delimiter for parsing the file
        
    Returns:
        List of SMILES strings
    """
    initial_smiles = []
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return initial_smiles
    
    with open(file_path, 'r') as f:
        # Skip header row
        header = next(f)
        
        # Read SMILES from first column
        for line in f:
            smiles = line.strip().split(delimiter)[0]
            if smiles:
                initial_smiles.append(smiles)
                
    print(f"Loaded {len(initial_smiles)} initial molecules from {file_path}")
    return initial_smiles

def is_valid_mol(smiles):
    """Check if a SMILES string represents a valid molecule.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Boolean indicating if molecule is valid
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    """Main function for molecule inference with LM-GAN."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up device and logging (only getting device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print arguments
    # print(args)
    
    # Turn off rdkit logging to reduce output
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    # Read configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Parse mutation parameters
    mutation_parameter_list = [float(x.strip()) for x in args.mutation_parameter.split(',')]
    
    # Initialize GAN operators with simpler arguments
    gan_args = argparse.Namespace()
    gan_args.generator_only = args.generator_only
    gan_args.lr = args.lr
    gan_args.top_k = args.top_k
    gan_args.random_init = False
    
    # Initialize GAN operators
    gan_operators = initialize_gan_operators(config, device, mutation_parameter_list, gan_args)
    
    # Load model weights if specified
    if args.model_file:
        print(f"Loading model weights from {args.model_file}")
        gan_operators[0]._gen.load_state_dict(torch.load(args.model_file, map_location=device))
        print("Model weights loaded successfully")
    
    # Initialize scoring operator
    scoring_operator = setup_scoring_operator(config, device)
    
    # Initialize scaffold handler if needed
    scaffold_handler = None
    if 'scaffold_operator' in config and args.use_scaffold:
        scaffold_handler = initialize_scaffold_handler(config, gan_operators, device)
    
    # Determine generation mode based on arguments
    if args.valid_unique_only:
        print("Generating valid unique molecules with scaffold")
        generation_mode = "valid_unique"
    elif args.valid_only:
        print("Generating valid molecules with scaffold (duplicates allowed)")
        generation_mode = "valid_only"
    else:
        print("Disabled scaffold validation for molecule generation")
        generation_mode = "no_validation"
    
    # Common setup for generation
    generated_smiles = []
    max_attempts = args.sample_size * 10  # Maximum number of attempts to prevent infinite loops
    attempt_count = 0
    
    # For valid_unique mode, we need to track unique molecules
    unique_smiles_set = set() if generation_mode == "valid_unique" else None
    
    # For all modes, we'll need to check if molecules are valid
    use_validation = generation_mode in ["valid_unique", "valid_only"]
    
    # Direct GAN generation without scaffold (if scaffold not used)
    if not args.use_scaffold or scaffold_handler is None:
        print("Generating molecules directly without scaffold...")
        
        # Use standard GAN molecule generation
        batch_size = args.batch_size
        
        # Load initial molecules from file or use defaults
        initial_smiles = []
       
        initial_smiles = load_initial_molecules(args.initial_molecules_file)
    
        while len(generated_smiles) < args.sample_size and attempt_count < max_attempts:
            attempt_count += 1
            
            # Randomly select a batch of molecules
            if len(initial_smiles) <= batch_size:
                current_batch = initial_smiles.copy()
            else:
                current_batch = random.sample(initial_smiles, batch_size)
            
            # Generate new molecules using the GAN
            new_smiles, _, _ = gan_operators[0].evaluate_generator(
                current_batch,
                use_scaffold=False,
                scaffold_handler=None,
                mask_mode=args.mask_mode
            )
                
            if generation_mode == "valid_unique":
                # Filter for valid molecules first if needed
                valid_smiles = [s for s in new_smiles if is_valid_mol(s)]
                
                # Add only unique valid molecules
                for smiles in valid_smiles:
                    if smiles not in unique_smiles_set:
                        unique_smiles_set.add(smiles)
                        generated_smiles.append(smiles)
                        
                        # If we have enough molecules, stop
                        if len(generated_smiles) >= args.sample_size:
                            break
            elif generation_mode == "valid_only":
                # Add all valid molecules (duplicates allowed)
                valid_molecules = [s for s in new_smiles if is_valid_mol(s)]
                generated_smiles.extend(valid_molecules)
                        
                # If we have enough molecules, stop
                if len(generated_smiles) >= args.sample_size:
                    break
            else:
                # For no_validation: add all molecules without checks
                generated_smiles.extend(new_smiles)
                
                # If we have enough molecules, stop
                if len(generated_smiles) >= args.sample_size:
                    break
            
            # Update initial_smiles for next iteration with newly generated molecules
            # This helps to generate more diverse molecules in subsequent iterations
            if new_smiles:
                initial_smiles = new_smiles
            
            # If we didn't get enough in this attempt, generate more
            if len(generated_smiles) < args.sample_size:
                mode_text = {
                    "valid_unique": "unique valid ",
                    "valid_only": "valid ",
                    "no_validation": ""
                }[generation_mode]
                
                print(f"Got {len(generated_smiles)}/{args.sample_size} {mode_text}molecules after attempt {attempt_count}/{max_attempts}, generating more...")
    
    # Scaffold-based generation
    else:
        print("Generating molecules using scaffold_handler...")
        
        while len(generated_smiles) < args.sample_size and attempt_count < max_attempts:
            attempt_count += 1
            
            # Generate a batch of molecules
            new_smiles = scaffold_handler.generate_initial_population(
                gan=gan_operators[0],
                population_size=args.batch_size,  # Small batch size helps to get higher validity
                batch_size=args.batch_size,
                top_k=args.top_k,
                use_simple_method=False,
                valid_only=use_validation  # Only validate if requested
            )
            
            if generation_mode == "valid_unique":
                # Add only unique valid molecules
                for smiles in new_smiles:
                    # Additional validation is already handled by scaffold_handler when valid_only=True
                    if smiles not in unique_smiles_set:
                        unique_smiles_set.add(smiles)
                        generated_smiles.append(smiles)
                        
                        # If we have enough molecules, stop
                        if len(generated_smiles) >= args.sample_size:
                            break
            else:
                # For valid_only or no_validation: add all molecules without uniqueness check
                generated_smiles.extend(new_smiles)
                
                # If we have enough molecules, stop
                if len(generated_smiles) >= args.sample_size:
                    break
            
            # If we didn't get enough in this attempt, generate more
            if len(generated_smiles) < args.sample_size:
                mode_text = {
                    "valid_unique": "unique valid ",
                    "valid_only": "valid ",
                    "no_validation": ""
                }[generation_mode]
                
                print(f"Got {len(generated_smiles)}/{args.sample_size} {mode_text}molecules after attempt {attempt_count}/{max_attempts}, generating more...")
    
    # If we reached max attempts but didn't get enough molecules, warn the user
    if attempt_count >= max_attempts and len(generated_smiles) < args.sample_size:
        mode_text = {
            "valid_unique": "unique valid ",
            "valid_only": "valid ",
            "no_validation": ""
        }[generation_mode]
        
        print(f"WARNING: Reached maximum number of attempts ({max_attempts}). Only generated {len(generated_smiles)}/{args.sample_size} {mode_text}molecules.")
    
    # Truncate to exactly the requested sample size or use all if we have fewer
    generated_smiles = generated_smiles[:args.sample_size]
    
    mode_text = {
        "valid_unique": "unique valid ",
        "valid_only": "valid ",
        "no_validation": ""
    }[generation_mode]
    
    scaffold_text = "with scaffold " if args.use_scaffold and scaffold_handler is not None else ""
    print(f"Generated {len(generated_smiles)} {mode_text}molecules {scaffold_text}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
    
    # Save generated SMILES to file
    with open(args.data_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['smiles'])  # Header
        for smiles in generated_smiles:
            writer.writerow([smiles])
    
    print(f"Saved {len(generated_smiles)} molecules to {args.data_file}")

if __name__ == "__main__":
    main() 