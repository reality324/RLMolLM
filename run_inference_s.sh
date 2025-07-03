#!/bin/bash

# =============================================
# EDITABLE PARAMETERS
# =============================================
# Define scaffold configurations
SCAFFOLD_KEYS=(
    # "scaffold_1_acry"
    # "scaffold_2_indole"
    # "scaffold_3_biphe"
    # "scaffold_4"
    "scaffold_5_male"
    "scaffold_6_benzene"
    "scaffold_7_dihydropyridine"
    "scaffold_8_benzothiophene"
)

# Define model types
MODEL_TYPES=(
    "alm_ppo"
    "alm"
    "lm_ppo"
    "lm_ng_ppo"
    "lm"
    "lm_ng"
)

# Define validation modes
VALIDATION_MODES=(
    "valid_unique_only"
    "valid_only"
    "no_validation"
)

# Sample size and repetitions
SAMPLE_SIZE=100
NUM_REPETITIONS=4

# Batch size for inference
BATCH_SIZE=32

# Top k sampling parameter
TOP_K=1

# =============================================
# SCRIPT EXECUTION (DO NOT EDIT BELOW)
# =============================================

# Create main log directory
LOG_DIR="./inference_logs"
mkdir -p "${LOG_DIR}"

# Main log file
MAIN_LOG="${LOG_DIR}/inference_main.log"
echo "Starting scaffold inference run at $(date)" > "${MAIN_LOG}"
echo "=======================================" >> "${MAIN_LOG}"

# Run inference for all configurations
for scaffold_key in "${SCAFFOLD_KEYS[@]}"; do
    echo "=============================================="
    echo "Running inference for ${scaffold_key}"
    echo "=============================================="
    echo "Running inference for ${scaffold_key}" >> "${MAIN_LOG}"
    
    # Create scaffold-specific output directory
    SCAFFOLD_OUTPUT_DIR="./inference_output/${scaffold_key}"
    mkdir -p "${SCAFFOLD_OUTPUT_DIR}"
    
    # Create scaffold-specific log file
    SCAFFOLD_LOG="${LOG_DIR}/${scaffold_key}.log"
    echo "Starting inference for ${scaffold_key} at $(date)" > "${SCAFFOLD_LOG}"
    echo "=======================================" >> "${SCAFFOLD_LOG}"
    
    for model_type in "${MODEL_TYPES[@]}"; do
        echo "----------------------------------------"
        echo "Running inference for model type: ${model_type}"
        echo "----------------------------------------"
        echo "Running inference for model type: ${model_type}" >> "${SCAFFOLD_LOG}"
        
        # Find best model file path
        MODEL_DIR="./training_output/${scaffold_key}/${model_type}_2000_t1_e20"
        
        # Check if model directory exists
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Model directory not found: ${MODEL_DIR}, skipping..."
            echo "Model directory not found: ${MODEL_DIR}, skipping..." >> "${SCAFFOLD_LOG}"
            continue
        fi
        
        # Find the best model file (based on epoch number)
        BEST_MODEL=$(find "$MODEL_DIR" -name "best_model_epoch_*_generator_0.pt" | sort -r | head -1)
        
        if [ -z "$BEST_MODEL" ]; then
            # If no best model found, try any model
            BEST_MODEL=$(find "$MODEL_DIR" -name "*_generator_0.pt" | sort -r | head -1)
        fi
        
        if [ -z "$BEST_MODEL" ]; then
            echo "No model file found in ${MODEL_DIR}, skipping..."
            echo "No model file found in ${MODEL_DIR}, skipping..." >> "${SCAFFOLD_LOG}"
            continue
        fi
        
        echo "Using model file: ${BEST_MODEL}"
        echo "Using model file: ${BEST_MODEL}" >> "${SCAFFOLD_LOG}"
        
        for validation_mode in "${VALIDATION_MODES[@]}"; do
            echo "Running with validation mode: ${validation_mode}"
            echo "Running with validation mode: ${validation_mode}" >> "${SCAFFOLD_LOG}"
            
            # Set validation flags and file suffix
            VALIDATION_FLAG=""
            FILE_SUFFIX=""
            
            if [ "$validation_mode" == "valid_unique_only" ]; then
                VALIDATION_FLAG="--valid_unique_only"
                FILE_SUFFIX="valid_unique_only"
            elif [ "$validation_mode" == "valid_only" ]; then
                VALIDATION_FLAG="--valid_only"
                FILE_SUFFIX="valid_only"
            else
                # For "no_validation" mode, use "any" in the filename
                FILE_SUFFIX="any"
            fi
            
            # Run multiple repetitions for error bar calculation
            for rep in $(seq 1 $NUM_REPETITIONS); do
                # Set output file with scaffold-specific directory and repetition number
                OUTPUT_FILE="${SCAFFOLD_OUTPUT_DIR}/${model_type}_${FILE_SUFFIX}_${rep}.csv"
                
                # Run inference
                echo "Generating molecules to ${OUTPUT_FILE} (sample size: ${SAMPLE_SIZE}, repetition: ${rep})"
                echo "Generating molecules to ${OUTPUT_FILE} (sample size: ${SAMPLE_SIZE}, repetition: ${rep})" >> "${SCAFFOLD_LOG}"
                
                # Run inference and capture output to log file
                {
                    python inference.py \
                        --config "./config/scaffold_examples/${scaffold_key}.json" \
                        --model_file "${BEST_MODEL}" \
                        --output_directory "${SCAFFOLD_OUTPUT_DIR}" \
                        --data_file "${OUTPUT_FILE}" \
                        --run_id "inference_${scaffold_key}_${model_type}_${rep}" \
                        --sample_size ${SAMPLE_SIZE} \
                        --batch_size ${BATCH_SIZE} \
                        --top_k ${TOP_K} \
                        --mask_mode sample_partition \
                        --use_scaffold \
                        ${VALIDATION_FLAG}
                } 2>&1 | tee -a "${SCAFFOLD_LOG}"
                
                echo "Completed generation for ${validation_mode} (repetition ${rep})"
                echo "Completed generation for ${validation_mode} (repetition ${rep})" >> "${SCAFFOLD_LOG}"
            done
            echo "----------------------------------------" >> "${SCAFFOLD_LOG}"
        done
        
        # Log completion of model type in main log
        echo "Completed inference for ${scaffold_key}/${model_type}" >> "${MAIN_LOG}"
    done
    
    echo "Completed all models for ${scaffold_key} at $(date)" >> "${SCAFFOLD_LOG}"
    echo "Completed all models for ${scaffold_key}" >> "${MAIN_LOG}"
    echo "----------------------------------------" >> "${MAIN_LOG}"
done

echo "All inference runs completed!"
echo "All inference runs completed at $(date)" >> "${MAIN_LOG}"
echo "=======================================" >> "${MAIN_LOG}" 