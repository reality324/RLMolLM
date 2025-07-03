#!/usr/bin/env python3
"""
Python script to run training for all configurations with different mutation parameters.
Equivalent to train_all_ns_random.sh but as a Python module for API integration.
"""

import os
import subprocess
import sys
from typing import Dict, List, Tuple
import argparse


class TrainingRunner:
    """Class to handle training execution with different configurations."""
    
    def __init__(self, base_scaffold: str = "no_scaffold_2_random", 
                 base_output_dir: str = "./training_output",
                 config_file: str = "./config/no_scaffold_2.json"):
        """
        Initialize the training runner.
        
        Args:
            base_scaffold: Base scaffold name for output
            base_output_dir: Base training output directory
            config_file: Config file to use
        """
        self.base_scaffold = base_scaffold
        self.base_output_dir = base_output_dir
        self.config_file = config_file
        
        # Initial population settings - use the original scaffold's population for new jobs
        if base_scaffold.startswith("job_"):
            # For job-specific scaffolds, use the original population
            population_scaffold = "no_scaffold_2_random"
        else:
            population_scaffold = base_scaffold
            
        self.initial_population_dir = f"{base_output_dir}/{population_scaffold}/2000_initial"
        self.initial_population_file = "initial_population.csv"
        self.initial_population = f"{self.initial_population_dir}/{self.initial_population_file}"
        
        # Mutation parameter cases to run
        # Format: (suffix, parameter) - suffix will be appended to output directory name
        self.mutation_cases = [
            ("_1m", 1),        # Default case (mutation parameter = 1)
            # ("_0p8m", 0.8),    # 0.8 mutation parameter case
            # ("_0p7m", 0.7),    # 0.7 mutation parameter case
            # ("_0p6m", 0.6),    # 0.6 mutation parameter case
            # ("_0p5m", 0.5),    # 0.5 mutation parameter case
        ]
        
        # Base common arguments shared by all configurations
        self.base_args = [
            "--population_size", "2000",
            "--mutation_samples", "2000",
            "--batch_size", "32",
            "--top_k", "1",
            "--init_top_k", "5",
            "--epochs", "50",
            "--lr", "0.00002",
            "--mask_mode", "random"
        ]
        
        # Define configurations
        self.configs = {
            "alm_ppo": "--generator_only --mlm_loss --use_ppo --reward_scale 1.5 --invalid_penalty -0.9",
            "alm": "--generator_only --mlm_loss",
            "lm_ppo": "--use_ppo --reward_scale 1.5 --invalid_penalty -0.9",
            "lm_ng_ppo": "--use_ppo --no_merge --reward_scale 1.5 --invalid_penalty -0.9",
            "lm": "",
            "lm_ng": "--no_merge"
        }
        
        # Define the order of configurations to run
        self.config_keys = [
            "alm_ppo",
            "alm",
            "lm_ppo", 
            "lm_ng_ppo",
            "lm",
            "lm_ng"
        ]
    
    def generate_initial_population(self) -> bool:
        """
        Generate initial population if it doesn't exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.initial_population is None:
            print("No initial population required - training will generate molecules")
            return True
            
        if os.path.exists(self.initial_population):
            print(f"Initial population already exists at: {self.initial_population}")
            return True
            
        print(f"Initial population file not found at {self.initial_population}")
        print("Generating initial population...")
        
        # Create initial population directory if it doesn't exist
        os.makedirs(self.initial_population_dir, exist_ok=True)
        
        # Generate initial population
        cmd = [
            "python", "generate_initial_population.py",
            "--output_file", self.initial_population,
            "--population_size", "2000",
            "--config", self.config_file
        ]
        
        try:
            # Change to project root directory for subprocess
            import pathlib
            project_root = pathlib.Path(__file__).parent.absolute()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
            print("Initial population generated successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to generate initial population")
            print(f"Command: {' '.join(cmd)}")
            print(f"Working directory: {pathlib.Path(__file__).parent.absolute()}")
            print(f"Error: {e.stderr}")
            return False
    
    def run_training_config(self, config_name: str, output_dir: str, 
                          mutation_parameter: float) -> bool:
        """
        Run training for a specific configuration.
        
        Args:
            config_name: Name of the configuration to run
            output_dir: Output directory for this training run
            mutation_parameter: Mutation parameter value
            
        Returns:
            bool: True if successful, False otherwise
        """
        config_args = self.configs[config_name]
        
        print("=====================================")
        print(f"Starting training: {config_name} for {self.base_scaffold}")
        print(f"Mutation parameter: {mutation_parameter}")
        print(f"Configuration: {config_args}")
        print("=====================================")
        
        # Build command
        cmd = [
            "python", "training_combined.py",
            "--output_directory", output_dir,
            "--config", self.config_file
        ]
        
        # Only add data_file if we have an initial population
        if self.initial_population is not None:
            cmd.extend(["--data_file", self.initial_population])
        else:
            print("No initial population specified - training will generate new one")
            
        cmd.extend(["--mutation_parameter", str(mutation_parameter)])
        
        # Add base arguments
        cmd.extend(self.base_args)
        
        # Add configuration-specific arguments
        if config_args:
            cmd.extend(config_args.split())
        
        try:
            # Change to project root directory for subprocess
            import pathlib
            project_root = pathlib.Path(__file__).parent.absolute()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
            print(f"Training completed for: {config_name}")
            print(f"Output saved to: {output_dir}")
            print("=====================================")
            print("")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Training failed for {config_name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Working directory: {pathlib.Path(__file__).parent.absolute()}")
            print(f"Exit code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    
    def run_mutation_case(self, suffix: str, parameter: float) -> bool:
        """
        Run all configurations for a specific mutation case.
        
        Args:
            suffix: Suffix for the mutation case (e.g., "_1m")
            parameter: Mutation parameter value
            
        Returns:
            bool: True if all configurations succeeded, False otherwise
        """
        # Set mutation folder name (without leading underscore)
        mutation_folder = suffix.lstrip('_')
        
        print("==============================================")
        print(f"Running training for {self.base_scaffold}/{mutation_folder} with mutation parameter {parameter}")
        print("==============================================")
        
        # Base directory for output for this scaffold variant
        scaffold_output_dir = f"{self.base_output_dir}/{self.base_scaffold}/{mutation_folder}"
        os.makedirs(scaffold_output_dir, exist_ok=True)
        
        # Generate initial population if needed (only if initial_population is set)
        if self.initial_population is not None:
            if not self.generate_initial_population():
                return False
            print(f"Using initial population from: {self.initial_population}")
        else:
            print("No initial population specified - training will generate molecules during training")
        
        # Run each configuration in order
        all_successful = True
        for config_name in self.config_keys:
            output_dir = f"{scaffold_output_dir}/{config_name}"
            success = self.run_training_config(config_name, output_dir, parameter)
            if not success:
                all_successful = False
                print(f"WARNING: Training failed for {config_name}, continuing with next configuration...")
        
        print(f"All training configurations completed for {self.base_scaffold}/{mutation_folder}!")
        print("")
        return all_successful
    
    def run_all(self) -> bool:
        """
        Run all mutation cases and configurations.
        
        Returns:
            bool: True if all runs succeeded, False otherwise
        """
        print("Starting training for all mutation cases...")
        
        all_successful = True
        for suffix, parameter in self.mutation_cases:
            success = self.run_mutation_case(suffix, parameter)
            if not success:
                all_successful = False
                print(f"WARNING: Some configurations failed for mutation case {suffix}")
        
        print("All training configurations completed for all mutation cases!")
        return all_successful
    
    def add_mutation_case(self, suffix: str, parameter: float):
        """Add a new mutation case to run."""
        self.mutation_cases.append((suffix, parameter))
    
    def remove_mutation_case(self, suffix: str):
        """Remove a mutation case by suffix."""
        self.mutation_cases = [(s, p) for s, p in self.mutation_cases if s != suffix]
    
    def enable_mutation_case(self, parameter: float):
        """Enable a mutation case by uncommenting it."""
        suffix_map = {
            0.8: "_0p8m",
            0.7: "_0p7m", 
            0.6: "_0p6m",
            0.5: "_0p5m"
        }
        if parameter in suffix_map:
            suffix = suffix_map[parameter]
            if (suffix, parameter) not in self.mutation_cases:
                self.mutation_cases.append((suffix, parameter))
    
    def set_config_keys(self, config_keys: List[str]):
        """Set which configurations to run."""
        self.config_keys = config_keys


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Run training for all configurations")
    parser.add_argument("--base_scaffold", default="no_scaffold_2_random",
                       help="Base scaffold name for output")
    parser.add_argument("--base_output_dir", default="./training_output",
                       help="Base training output directory")
    parser.add_argument("--config_file", default="./config/no_scaffold_2.json",
                       help="Config file to use")
    parser.add_argument("--mutation_params", nargs="+", type=float, default=[1],
                       help="List of mutation parameters to run")
    parser.add_argument("--configs", nargs="+", 
                       choices=["alm_ppo", "alm", "lm_ppo", "lm_ng_ppo", "lm", "lm_ng"],
                       default=["alm_ppo", "alm", "lm_ppo", "lm_ng_ppo", "lm", "lm_ng"],
                       help="Which configurations to run")
    
    args = parser.parse_args()
    
    # Create runner
    runner = TrainingRunner(
        base_scaffold=args.base_scaffold,
        base_output_dir=args.base_output_dir, 
        config_file=args.config_file
    )
    
    # Set custom mutation cases if provided
    if args.mutation_params != [1]:
        runner.mutation_cases = []
        for param in args.mutation_params:
            if param == 1:
                suffix = "_1m"
            else:
                suffix = f"_{str(param).replace('.', 'p')}m"
            runner.mutation_cases.append((suffix, param))
    
    # Set custom configs if provided
    runner.set_config_keys(args.configs)
    
    # Run all training
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 