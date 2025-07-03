#!/bin/bash

# Editable parameters
BASE_SCAFFOLD="no_scaffold_2_random"
TEMP_OUTPUT_DIR="./training_output/${BASE_SCAFFOLD}/temp_initial"
FINAL_OUTPUT_DIR="./training_output/${BASE_SCAFFOLD}/2000_initial"
TEMP_DATA_FILE="${TEMP_OUTPUT_DIR}/initial_population.csv"
FINAL_DATA_FILE="${FINAL_OUTPUT_DIR}/initial_population.csv"

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
    # --use_ppo
    # --reward_scale 1.5
    # --invalid_penalty -0.9
    # --use_scaffold
    --mask_mode random
    --config ./config/no_scaffold_2_random.json
    # --data_file "${DATA_FILE}"  # Use the editable data file path
)

echo "Starting initial population generation without scaffolds..."
echo "Base scaffold: ${BASE_SCAFFOLD}"
echo "Temporary output: ${TEMP_OUTPUT_DIR}"
echo "Final output: ${FINAL_OUTPUT_DIR}"

# Create temporary output directory
mkdir -p "${TEMP_OUTPUT_DIR}"

# Run the training command to generate initial population
python training_combined.py \
    --output_directory "${TEMP_OUTPUT_DIR}" \
    "${COMMON_ARGS[@]}"

# Check if initial population was generated successfully
if [ -f "${TEMP_DATA_FILE}" ]; then
    echo "Initial population generated successfully!"
    
    # Create final output directory
    mkdir -p "${FINAL_OUTPUT_DIR}"
    
    # Move the initial population file to the correct location
    mv "${TEMP_DATA_FILE}" "${FINAL_DATA_FILE}"
    
    echo "Moved initial population to: ${FINAL_DATA_FILE}"
    
    # Clean up temporary directory
    rm -rf "${TEMP_OUTPUT_DIR}"
    
    echo "Initial population generation completed successfully!"
    echo "Final output saved to: ${FINAL_DATA_FILE}"
    
    # Display file info
    if [ -f "${FINAL_DATA_FILE}" ]; then
        echo "Generated $(wc -l < "${FINAL_DATA_FILE}") molecules (including header)"
        echo "File size: $(du -h "${FINAL_DATA_FILE}" | cut -f1)"
    fi
else
    echo "ERROR: Initial population file not found at ${TEMP_DATA_FILE}"
    echo "Training may have failed. Please check the output above for errors."
    exit 1
fi 