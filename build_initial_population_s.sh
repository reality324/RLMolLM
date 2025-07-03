#!/bin/bash

# Set common parameters
COMMON_ARGS=(
    --population_size 2000
    --mutation_samples 2000
    --mutation_parameter 1
    --batch_size 32
    --top_k 1
    --init_top_k 5
    --epochs 0
    --lr 0.00002
    --generator_only
    --mlm_loss
    --use_ppo
    --reward_scale 1.5
    --invalid_penalty -0.9
    --use_scaffold
    --mask_mode sample_partition
)

# Define scaffold configurations in order
SCAFFOLD_KEYS=(
    "scaffold_6_benzene"
    # "scaffold_7_dihydropyridine"
    # "scaffold_8_benzothiophene"
)

# Define scaffold names
declare -A SCAFFOLD_NAMES=(
    ["scaffold_1_acry"]="Acrylamide"
    ["scaffold_2_indole"]="Indole"
    ["scaffold_3_biphe"]="Biphenyl"
    ["scaffold_4"]="Scaffold_4"
    ["scaffold_5_male"]="Male"
    ["scaffold_6_benzene"]="Benzene"
    ["scaffold_7_dihydropyridine"]="dihydropyridine"
    ["scaffold_8_benzothiophene"]="benzothiophene"
)

# Run training for each scaffold in order
for scaffold_key in "${SCAFFOLD_KEYS[@]}"; do
    scaffold_name=${SCAFFOLD_NAMES[$scaffold_key]}
    echo "Starting initial population generation for ${scaffold_name}..."
    
    # Define directories
    TEMP_OUTPUT_DIR="./training_output/${scaffold_key}/temp_initial"
    FINAL_OUTPUT_DIR="./training_output/${scaffold_key}/2000_initial"
    TEMP_DATA_FILE="${TEMP_OUTPUT_DIR}/initial_population.csv"
    FINAL_DATA_FILE="${FINAL_OUTPUT_DIR}/initial_population.csv"
    
    echo "Temporary output: ${TEMP_OUTPUT_DIR}"
    echo "Final output: ${FINAL_OUTPUT_DIR}"
    
    # Create temporary output directory
    mkdir -p "${TEMP_OUTPUT_DIR}"
    
    # Run the training command to generate initial population
    python training_combined.py \
        --config ./config/scaffold_examples/${scaffold_key}.json \
        --output_directory "${TEMP_OUTPUT_DIR}" \
        "${COMMON_ARGS[@]}"
    
    # Check if initial population was generated successfully
    if [ -f "${TEMP_DATA_FILE}" ]; then
        echo "Initial population generated successfully for ${scaffold_name}!"
        
        # Create final output directory
        mkdir -p "${FINAL_OUTPUT_DIR}"
        
        # Move the initial population file to the correct location
        mv "${TEMP_DATA_FILE}" "${FINAL_DATA_FILE}"
        
        echo "Moved initial population to: ${FINAL_DATA_FILE}"
        
        # Clean up temporary directory
        rm -rf "${TEMP_OUTPUT_DIR}"
        
        echo "Initial population generation completed for ${scaffold_name}!"
        
        # Display file info
        if [ -f "${FINAL_DATA_FILE}" ]; then
            echo "Generated $(wc -l < "${FINAL_DATA_FILE}") molecules (including header)"
            echo "File size: $(du -h "${FINAL_DATA_FILE}" | cut -f1)"
        fi
    else
        echo "ERROR: Initial population file not found at ${TEMP_DATA_FILE}"
        echo "Training may have failed for ${scaffold_name}. Please check the output above for errors."
        # Continue with next scaffold instead of exiting
        echo "Continuing with next scaffold..."
    fi
    
    echo "Completed initial population generation for ${scaffold_name}"
    echo "----------------------------------------"
done

echo "All initial population generation runs completed!" 