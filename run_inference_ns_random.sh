#!/bin/bash

#====================================#
# CONFIGURABLE VARIABLES (Edit here) #
#====================================#

# Base scaffold name
BASE_SCAFFOLD="no_scaffold_2_random"

# Mutation parameter cases to run (each will create a separate output directory)
# Format: "suffix:parameter" - suffix will be appended to output directory name
MUTATION_CASES=(
    "_1m:1"        # Default case (mutation parameter = 1)
    # "_0p7m:0.7"    # 0.7 mutation parameter case
)

# Number of samples per method (for error bar calculation)
NUM_SAMPLES=4

# Base paths
BASE_OUTPUT_DIR="./inference_output"
BASE_TRAINING_DIR="./training_output"

# Config file to use
CONFIG_FILE="./config/no_scaffold_2.json"

# UMAP visualization parameters (these will be used by the analysis script)
# These parameters can be modified to adjust the UMAP visualization
# UMAP_N_NEIGHBORS="10"     # Default: 5 (Higher values (10-50) emphasize global structure)
# UMAP_MIN_DIST="0.5"       # Default: 0.1 (Lower values (0.0-0.3) make tighter clusters)
# UMAP_METRIC="jaccard"    # Distance metric for molecular fingerprints

# Common parameters
BATCH_SIZE=32
TOP_K=1 # changed from 1
MASK_MODE="random"

# Model types to run inference for
MODEL_TYPES=(
    "alm_ppo"
    "alm"
    "lm_ppo"
    "lm_ng_ppo"
    "lm"
    "lm_ng"
)

# Validation modes to use
VALIDATION_MODES=(
    "valid_unique_only"
    "valid_only"
    "no_validation"
)

# Sample sizes - different for each validation mode
SAMPLE_SIZE_VALID_UNIQUE=100
SAMPLE_SIZE_VALID_ONLY=100
SAMPLE_SIZE_ANY=5000

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
    echo "Running inference for ${BASE_SCAFFOLD}/${MUTATION_FOLDER} with mutation parameter ${parameter}"
    echo "=============================================="
    
    # Define output directory for this case
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BASE_SCAFFOLD}/${MUTATION_FOLDER}"
    mkdir -p "${OUTPUT_DIR}"
    
    # Save UMAP parameters for later analysis
    UMAP_PARAMS_FILE="${OUTPUT_DIR}/umap_params.txt"
    echo "UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS}" > "${UMAP_PARAMS_FILE}"
    echo "UMAP_MIN_DIST=${UMAP_MIN_DIST}" >> "${UMAP_PARAMS_FILE}"
    echo "UMAP_METRIC=${UMAP_METRIC}" >> "${UMAP_PARAMS_FILE}"
    
    # Define log file
    LOG_FILE="${OUTPUT_DIR}/inference_log.txt"
    echo "Starting inference run for ${BASE_SCAFFOLD}/${MUTATION_FOLDER} at $(date)" > "${LOG_FILE}"
    echo "Mutation parameter: ${parameter}" >> "${LOG_FILE}"
    echo "Number of samples per method: ${NUM_SAMPLES}" >> "${LOG_FILE}"
    echo "UMAP parameters: neighbors=${UMAP_N_NEIGHBORS}, min_dist=${UMAP_MIN_DIST}, metric=${UMAP_METRIC}" >> "${LOG_FILE}"
    echo "=======================================" >> "${LOG_FILE}"
    
    # Define initial molecules file - from the base scaffold common directory
    INITIAL_MOLECULES="${BASE_TRAINING_DIR}/${BASE_SCAFFOLD}/2000_initial/initial_population.csv"
    
    # Check if initial molecules file exists
    if [ ! -f "${INITIAL_MOLECULES}" ]; then
        echo "ERROR: Initial molecules file not found at ${INITIAL_MOLECULES}"
        echo "Please ensure the file exists or adjust the path in the script."
        echo "ERROR: Initial molecules file not found at ${INITIAL_MOLECULES}" >> "${LOG_FILE}"
        continue
    fi
    
    echo "Using initial molecules from: ${INITIAL_MOLECULES}" | tee -a "${LOG_FILE}"
    
    echo "Running inference for ${BASE_SCAFFOLD}/${MUTATION_FOLDER} models" | tee -a "${LOG_FILE}"
    
    for model_type in "${MODEL_TYPES[@]}"; do
        echo "----------------------------------------"
        echo "Running inference for model type: ${model_type}"
        echo "----------------------------------------"
        echo "----------------------------------------" >> "${LOG_FILE}"
        echo "Running inference for model type: ${model_type}" >> "${LOG_FILE}"
        
        # Find best model file path in the new directory structure
        MODEL_DIR="${BASE_TRAINING_DIR}/${BASE_SCAFFOLD}/${MUTATION_FOLDER}/${model_type}"
        
        echo "Looking for model directory: ${MODEL_DIR}"
        
        # If folder doesn't exist, try alternative directory structure (model_2000_t1_e20)
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Directory not found, trying alternative format..."
            MODEL_DIR="${BASE_TRAINING_DIR}/${BASE_SCAFFOLD}/${MUTATION_FOLDER}/${model_type}_2000_t1_e20"
            echo "Looking for model directory: ${MODEL_DIR}"
        fi
        
        # Check if model directory exists
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Model directory not found: ${MODEL_DIR}, skipping..."
            echo "Model directory not found: ${MODEL_DIR}, skipping..." >> "${LOG_FILE}"
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
            echo "No model file found in ${MODEL_DIR}, skipping..." >> "${LOG_FILE}"
            continue
        fi
        
        echo "Using model file: ${BEST_MODEL}"
        echo "Using model file: ${BEST_MODEL}" >> "${LOG_FILE}"
        
        for validation_mode in "${VALIDATION_MODES[@]}"; do
            echo "Running with validation mode: ${validation_mode}"
            echo "Running with validation mode: ${validation_mode}" >> "${LOG_FILE}"
            
            # Set validation flags, file suffix, and sample size based on validation mode
            VALIDATION_FLAG=""
            FILE_SUFFIX=""
            SAMPLE_SIZE=""
            
            if [ "$validation_mode" == "valid_unique_only" ]; then
                VALIDATION_FLAG="--valid_unique_only"
                FILE_SUFFIX="valid_unique_only"
                SAMPLE_SIZE=${SAMPLE_SIZE_VALID_UNIQUE}
            elif [ "$validation_mode" == "valid_only" ]; then
                VALIDATION_FLAG="--valid_only"
                FILE_SUFFIX="valid_only"
                SAMPLE_SIZE=${SAMPLE_SIZE_VALID_ONLY}
            else
                # For "no_validation" mode, use "any" in the filename
                FILE_SUFFIX="any"
                SAMPLE_SIZE=${SAMPLE_SIZE_ANY}
            fi
            
            echo "Using sample size: ${SAMPLE_SIZE} for validation mode: ${validation_mode}"
            echo "Using sample size: ${SAMPLE_SIZE} for validation mode: ${validation_mode}" >> "${LOG_FILE}"
            
            # Run multiple samples for error bar calculation
            for sample_idx in $(seq 1 $NUM_SAMPLES); do
                # Set output file with sample index
                OUTPUT_FILE="${OUTPUT_DIR}/${model_type}_${FILE_SUFFIX}_${sample_idx}.csv"
                
                # Run inference
                echo "Generating molecules to ${OUTPUT_FILE} (sample size: ${SAMPLE_SIZE}, sample: ${sample_idx}/${NUM_SAMPLES})"
                echo "Generating molecules to ${OUTPUT_FILE} (sample size: ${SAMPLE_SIZE}, sample: ${sample_idx}/${NUM_SAMPLES})" >> "${LOG_FILE}"
                
                # Run inference and capture output to log file
                {
                    python inference.py \
                        --config "${CONFIG_FILE}" \
                        --model_file "${BEST_MODEL}" \
                        --output_directory "${OUTPUT_DIR}" \
                        --data_file "${OUTPUT_FILE}" \
                        --initial_molecules_file "${INITIAL_MOLECULES}" \
                        --run_id "inference_${BASE_SCAFFOLD}_${MUTATION_FOLDER}_${model_type}_${sample_idx}" \
                        --sample_size ${SAMPLE_SIZE} \
                        --batch_size ${BATCH_SIZE} \
                        --mutation_parameter ${parameter} \
                        --top_k ${TOP_K} \
                        --mask_mode ${MASK_MODE} \
                        ${VALIDATION_FLAG}
                } 2>&1 | tee -a "${LOG_FILE}"
                
                echo "Completed generation for ${validation_mode} (sample: ${sample_idx}/${NUM_SAMPLES})"
                echo "Completed generation for ${validation_mode} (sample: ${sample_idx}/${NUM_SAMPLES})" >> "${LOG_FILE}"
            done
            
            echo "Completed all samples for ${validation_mode}"
            echo "Completed all samples for ${validation_mode}" >> "${LOG_FILE}"
            echo "----------------------------------------" >> "${LOG_FILE}"
        done
    done
    
    echo "Inference runs completed for ${BASE_SCAFFOLD}/${MUTATION_FOLDER}!"
    echo "Inference runs completed for ${BASE_SCAFFOLD}/${MUTATION_FOLDER} at $(date)" >> "${LOG_FILE}"
    echo "=======================================" >> "${LOG_FILE}"
done

echo "All inference runs completed for all mutation cases!" 