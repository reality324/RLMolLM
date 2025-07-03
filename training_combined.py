# training_combined.py
import torch
import numpy as np
import json
import rdkit.Chem
from rdkit import RDLogger
import os
import sys
import argparse
import random
from population.population import Population
from utils.util import (
    setup_logging,
    parse_mutation_params,
    parse_train_flags,
    setup_device_and_logging,
    initialize_gan_operators,
    initialize_scaffold_handler,
    generate_initial_population_from_scaffold,
    generate_initial_population_without_scaffold,
    save_models,
)
from utils.training_utils import (
    train_and_evolve,
    setup_scoring_operator,
)

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set seed for RDKit if possible
    try:
        rdkit.Chem.SetRandomSeed(seed)
    except:
        print(f"Warning: Could not set RDKit random seed. This might affect reproducibility.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LM-GAN for molecule generation with scaffolds')

    # Input/Output
    parser.add_argument('--config', type=str, default='./config/default.json', 
                        help='json config for scoring operators and gans')
    parser.add_argument('--data_file', type=str, default=None, 
                        help='file with smiles data in first column (optional)')
    parser.add_argument('--output_directory', type=str, default='./output', 
                        help='output directory for population and new sequences')
    parser.add_argument('--run_id', type=str, default='run', 
                        help='run id used as a prefix in output files')
    parser.add_argument('--log_file', type=str, default=None, 
                        help='path to log file (if not specified, logs to stdout)')

    # Environment
    parser.add_argument('--use_mpi', action='store_true', default=False, 
                        help='option to use mpi to determine device')
    parser.add_argument('--data_file_postfix', action='store_true', default=False, 
                        help='option to use different postfix based on rank for data file')
    parser.add_argument('--data_file_tag', type=str, default=None, 
                        help='tag for data file to be used with data_file_postfix option')
    # Add random seed argument
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for reproducibility')

    # Hyperparameters
    parser.add_argument('--population_size', type=int, default=1000, 
                        help='maximum number of training samples to use')
    parser.add_argument('--mutation_samples', type=str, default='1000', 
                        help='number of samples to be sent to the generator for evaluation')
    parser.add_argument('--mutation_parameter', type=str, default='0.15', 
                        help='determines fraction of tokens that are masked for generator')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size for model training')
    # Add to the parser arguments
    parser.add_argument('--top_k', type=int, default=5, 
                        help='number of top predictions for generator evaluation during training')
    parser.add_argument('--init_top_k', type=int, default=1, 
                        help='number of top predictions for initial population generation')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.00005, 
                        help='learning rate for model training')
    
    # Behavior flags
    parser.add_argument('--generator_only', action='store_true', default=False, 
                        help='option to turn off GAN training')
    parser.add_argument('--no_merge', action='store_true', default=False, 
                        help='option to turn off child population merge')
    parser.add_argument('--add_randomized_smiles', action='store_true', default=False, 
                        help='option to add randomized smiles for generator evaluation')
    parser.add_argument('--random_init', action='store_true', default=False, 
                        help='option to use random weights to intialize language model')
    parser.add_argument('--reset_all_smiles', action='store_true', default=False, 
                        help='option to count novel based only on current population')
    parser.add_argument('--mlm_loss', action='store_true', default=False, 
                        help='option to use mlm loss to train generator only')
    parser.add_argument('--train_flags', type=str, default=None, 
                        help='comma separated list of 0 or 1 to determine if gan is trained')
    
    parser.add_argument('--use_reinforce', action='store_true', default=False, 
                    help='option to use REINFORCE training after GAN/MLM training')
    
    # # Scaffold options
    parser.add_argument('--use_scaffold', action='store_true', default=False,
                       help='option to use scaffold-based generation')
    
    ##
    # PPO arguments
    parser.add_argument('--use_ppo', action='store_true', default=False, 
                        help='option to use PPO training')
    parser.add_argument('--ppo_epochs', type=int, default=1, #2 is worse 
                        help='number of PPO epochs per training batch')
    parser.add_argument('--ppo_interval', type=int, default=1, 
                        help='run PPO training every N epochs (default: every epoch)')
    parser.add_argument('--clip_ratio', type=float, default=0.2, 
                        help='PPO clip ratio')
    parser.add_argument('--entropy_coef', type=float, default=0.01, 
                        help='entropy coefficient for PPO loss')
    parser.add_argument('--value_coef', type=float, default=0.5, 
                        help='value loss coefficient for PPO loss')
    parser.add_argument('--reward_scale', type=float, default=1.0, 
                        help='scaling factor for rewards in PPO')
    parser.add_argument('--invalid_penalty', type=float, default=-1, 
                        help='penalty for invalid molecules in PPO')
    # PPO arguments


    parser.add_argument('--mask_mode', type=str, default='replace',
                      choices=['replace', 'random', 'sample_partition', 'pure_random_mask'],
                      help='Masking mode for generator evaluation')

    return parser.parse_args()

def main():
    """Main function for molecule generation with LM-GAN."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Set up MPI environment if enabled
    device, run_id, data_file, log_file = setup_device_and_logging(args)
    
    # Set up logging if not already done
    if log_file is None:
        log_file = setup_logging(args)
    
    # Print arguments and random seed info
    print(args, file=log_file, flush=True)
    print(f"Using random seed: {args.seed} for reproducibility", file=log_file, flush=True)
    
    # Turn off rdkit logging to reduce output
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    # Read configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Parse mutation parameters
    mutation_samples_list, mutation_parameter_list = parse_mutation_params(
        args.mutation_samples, 
        args.mutation_parameter
    )
    
    # Parse train flags
    train_flags = parse_train_flags(args.train_flags)
    if train_flags is not None and len(train_flags) != len(config['gan_operators']):
        print('train_flags must be same length as number of gans', file=log_file, flush=True)
        sys.exit(1)
    
    # Initialize GAN operators
    gan_operators = initialize_gan_operators(config, device, mutation_parameter_list, args)
    
    # Initialize scoring operator
    scoring_operator = setup_scoring_operator(config, device)
    
    # Initialize scaffold handler if needed
    scaffold_handler = None
    if 'scaffold_operator' in config:
        scaffold_handler = initialize_scaffold_handler(config, gan_operators, device)
    
    # Handle initial population generation
    if data_file is None:
        if args.use_scaffold and scaffold_handler is not None:
            # Generate scaffold-based population
            data_file = generate_initial_population_from_scaffold(
                scaffold_handler=scaffold_handler,
                gan_operators=gan_operators,
                args=args,
                log_file=log_file
            )
        else:
            # Generate population without scaffold
            print("Generating initial population without scaffold...", file=log_file, flush=True)
            
            # Extract min_masks and max_masks from config if available
            min_masks = 3  # Default value
            max_masks = 25  # Default value
            if 'no_scaffold_operator' in config:
                min_masks = config['no_scaffold_operator'].get('min_masks', min_masks)
                max_masks = config['no_scaffold_operator'].get('max_masks', max_masks)
                print(f"Using min_masks={min_masks}, max_masks={max_masks} from config", file=log_file, flush=True)
            
            # Pass min_masks and max_masks to the generate_initial_population_without_scaffold function
            data_file = generate_initial_population_without_scaffold(
                gan_operators=gan_operators,
                population_size=args.population_size,
                batch_size=10,  # Use default batch size
                top_k=args.init_top_k,  # Use init_top_k from args
                log_file=log_file,
                output_directory=args.output_directory,
                min_masks=min_masks,
                max_masks=max_masks
            )
    
    # Initialize population
    population = Population(
        gan_operators, 
        scoring_operator,
        use_scaffold=args.use_scaffold,
        scaffold_handler=scaffold_handler,
        mask_mode=args.mask_mode
    )
    # Load initial population
    if data_file is not None:
        population.read_population_dict_from_file(data_file, args.population_size)
        all_smiles = set(population.population_sequences)
        print(f'Data Samples: {len(all_smiles)}, Population size: {population.population_size}', 
              file=log_file, flush=True)
    else:
        print('No data file provided and generation not enabled.', 
              file=log_file, flush=True)
        sys.exit(1)
    
    # Train models and evolve population
    population, all_smiles = train_and_evolve(
        population=population,
        gan_operators=gan_operators,
        args=args, 
        train_flags=train_flags,
        all_smiles=all_smiles,
        log_file=log_file,
        scaffold_handler=scaffold_handler,
        use_scaffold=args.use_scaffold
    )
    
    # Save models
    if args.output_directory is not None:
        save_models(
            gan_operators=gan_operators,
            output_directory=args.output_directory,
            run_id=run_id,
            log_file=log_file
        )
    
    # Close log file
    if log_file is not None:
        log_file.close()

if __name__ == "__main__":
    main()