#!/bin/bash

# Define scaffold configurations in order
SCAFFOLD_KEYS=(
    "scaffold_6_benzene"
    # "scaffold_7_dihydropyridine"
    # "scaffold_8_benzothiophene"
)

# Define scaffold names
declare -A SCAFFOLD_NAMES=(
    ["scaffold_6_benzene"]="Benzene"
    ["scaffold_7_dihydropyridine"]="dihydropyridine"
    ["scaffold_8_benzothiophene"]="benzothiophene"
)

# Base common parameters
BASE_ARGS=(
    --population_size 2000
    --mutation_samples 2000
    --mutation_parameter 1
    --batch_size 32
    --top_k 1
    --init_top_k 5
    --epochs 20
    --lr 0.00002
    --use_scaffold
    --mask_mode sample_partition
)

# Define training methods
declare -A TRAINING_METHODS=(
    ["alm_ppo"]="ALM-RL (PPO): generator_only + mlm_loss + ppo;--generator_only --mlm_loss --use_ppo --reward_scale 1.5 --invalid_penalty -0.9"
    ["alm"]="ALM: generator_only + mlm_loss;--generator_only --mlm_loss"
    ["lm_ppo"]="LM-RL (PPO): ppo only;--use_ppo --reward_scale 1.5 --invalid_penalty -0.9"
    ["lm_ng_ppo"]="LM-NG-RL (PPO): ppo + no_merge;--use_ppo --reward_scale 1.5 --invalid_penalty -0.9 --no_merge"
    ["lm"]="LM: standard;--"
    ["lm_ng"]="LM-NG: no_merge;--no_merge"
)

# Run all training configurations for each scaffold
for scaffold_key in "${SCAFFOLD_KEYS[@]}"; do
    scaffold_name=${SCAFFOLD_NAMES[$scaffold_key]}
    echo "=============================================="
    echo "Starting all training configurations for ${scaffold_name}..."
    echo "=============================================="
    
    # Path to the data file
    DATA_FILE="./training_output/${scaffold_key}/2000_initial/initial_population.csv"
    
    # Loop through all training methods
    for method_key in "${!TRAINING_METHODS[@]}"; do
        IFS=';' read -r method_description method_args <<< "${TRAINING_METHODS[$method_key]}"
        
        echo "Running ${method_description} for ${scaffold_name}..."
        
        # Create command with proper arguments
        cmd="python training_combined.py \
            --config ./config/scaffold_examples/${scaffold_key}.json \
            --data_file \"${DATA_FILE}\" \
            --output_directory ./training_output/${scaffold_key}/${method_key}_2000_t1_e20"
        
        # Add method-specific arguments if they exist
        if [ "$method_args" != "--" ]; then
            cmd="${cmd} ${method_args}"
        fi
        
        # Add base arguments
        cmd="${cmd} ${BASE_ARGS[@]}"
        
        # Execute the command
        eval $cmd
        
        echo "Completed ${method_description} for ${scaffold_name}"
        echo "----------------------------------------"
    done
    
    echo "Completed all training configurations for ${scaffold_name}"
    echo "=============================================="
done

echo "All training runs completed!" 