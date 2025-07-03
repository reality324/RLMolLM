#!/bin/bash

#====================================#
# CONFIGURABLE VARIABLES (Edit here) #
#====================================#

# Base scaffold name for output
BASE_SCAFFOLD="no_scaffold_2_random"

# Base training output directory
BASE_OUTPUT_DIR="./training_output"

# Initial population directory and file
INITIAL_POPULATION_DIR="${BASE_OUTPUT_DIR}/${BASE_SCAFFOLD}/2000_initial"
INITIAL_POPULATION_FILE="initial_population.csv"
INITIAL_POPULATION="${INITIAL_POPULATION_DIR}/${INITIAL_POPULATION_FILE}"

# Mutation parameter cases to run (each will create a separate output directory)
# Format: "suffix:parameter" - suffix will be appended to output directory name
MUTATION_CASES=(
    "_1m:1"        # Default case (mutation parameter = 1)
    # "_0p7m:0.7"    # 0.7 mutation parameter case
)

# Config file to use
CONFIG_FILE="./config/no_scaffold_2_random.json"

# Base common arguments shared by all configurations
BASE_ARGS=(
    --population_size 2000
    --mutation_samples 2000
    --batch_size 32
    --top_k 1
    --init_top_k 5
    --epochs 50
    --lr 0.00002
    # --use_scaffold  # Non-scaffold mode
    --mask_mode random
)

# Define configurations as associative array
declare -A CONFIGS=(
    ["alm_ppo"]="--generator_only --mlm_loss --use_ppo --reward_scale 1.5 --invalid_penalty -0.9"
    ["alm"]="--generator_only --mlm_loss"
    ["lm_ppo"]="--use_ppo --reward_scale 1.5 --invalid_penalty -0.9"
    ["lm_ng_ppo"]="--use_ppo --no_merge --reward_scale 1.5 --invalid_penalty -0.9"
    ["lm"]=""
    ["lm_ng"]="--no_merge"
)

# Define the order of configurations to run
CONFIG_KEYS=(
    "alm_ppo"
    "alm"
    "lm_ppo"
    "lm_ng_ppo"
    "lm"
    "lm_ng"
)

#====================================#
# SCRIPT EXECUTION (Don't edit      #
# unless you know what you're doing) #
#====================================#

# Process each mutation case
for mutation_case in "${MUTATION_CASES[@]}"; do
    # Split the case into suffix and parameter
    IFS=':' read -r suffix parameter <<< "${mutation_case}"
    
    # Set mutation folder name (without appending to scaffold name)
    MUTATION_FOLDER="${suffix#_}"  # Remove the leading underscore from suffix
    
    echo "=============================================="
    echo "Running training for ${BASE_SCAFFOLD}/${MUTATION_FOLDER} with mutation parameter ${parameter}"
    echo "=============================================="
    
    # Base directory for output for this scaffold variant
    SCAFFOLD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BASE_SCAFFOLD}/${MUTATION_FOLDER}"
    mkdir -p "${SCAFFOLD_OUTPUT_DIR}"
    
    # Check if initial population file exists
    if [ ! -f "${INITIAL_POPULATION}" ]; then
        echo "Initial population file not found at ${INITIAL_POPULATION}"
        echo "Generating initial population..."
        
        # Create initial population directory if it doesn't exist
        mkdir -p "${INITIAL_POPULATION_DIR}"
        
        # Generate initial population
        python generate_initial_population.py \
            --output_file "${INITIAL_POPULATION}" \
            --population_size 2000 \
            --config "${CONFIG_FILE}"
        
        if [ ! -f "${INITIAL_POPULATION}" ]; then
            echo "ERROR: Failed to generate initial population at ${INITIAL_POPULATION}"
            echo "Please ensure the generate_initial_population.py script exists and works correctly."
            continue
        fi
    fi
    
    echo "Using initial population from: ${INITIAL_POPULATION}"
    
    # Run each configuration in order
    for config_name in "${CONFIG_KEYS[@]}"; do
        config_args=${CONFIGS[$config_name]}
        output_dir="${SCAFFOLD_OUTPUT_DIR}/${config_name}"
        
        echo "====================================="
        echo "Starting training: ${config_name} for ${BASE_SCAFFOLD}/${MUTATION_FOLDER}"
        echo "Mutation parameter: ${parameter}"
        echo "Configuration: ${config_args}"
        echo "====================================="
        
        # Convert string to array
        IFS=' ' read -r -a args_array <<< "${config_args}"
        
        # Run the training
        python training_combined.py \
            --output_directory "${output_dir}" \
            --config "${CONFIG_FILE}" \
            --data_file "${INITIAL_POPULATION}" \
            --mutation_parameter "${parameter}" \
            "${BASE_ARGS[@]}" \
            "${args_array[@]}"
        
        echo "Training completed for: ${config_name} (${BASE_SCAFFOLD}/${MUTATION_FOLDER})"
        echo "Output saved to: ${output_dir}"
        echo "====================================="
        echo ""
    done
    
    echo "All training configurations completed for ${BASE_SCAFFOLD}/${MUTATION_FOLDER}!"
    echo ""
done

echo "All training configurations completed for all mutation cases!" 