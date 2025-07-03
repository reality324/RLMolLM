#!/bin/bash

#====================================#
# CONFIGURABLE VARIABLES (Edit here) #
#====================================#

# Base scaffold name
BASE_SCAFFOLD="no_scaffold_2_random"

# Mutation folders to analyze (subfolders under BASE_SCAFFOLD)
MUTATION_FOLDERS=(
    "1m"        # Default case (mutation parameter = 1)
    # "0p8m"      # 0.8 mutation parameter case
    # "0p7m"      # 0.7 mutation parameter case
    # "0p6m"      # 0.6 mutation parameter case
    # "0p5m"      # 0.5 mutation parameter case
)

# Number of samples per method (for error bar calculation)
# Note: Only the first sample (_1) will be used for UMAP visualization
# All samples will be used to calculate error bars for validity and uniqueness
# and for averaging property curves
NUM_SAMPLES=8

# Base directories
ANALYSIS_DIR="./analysis/plots/${BASE_SCAFFOLD}"
LOGS_DIR="./analysis/logs"

# Input directory pattern (where generated molecules are stored)
INPUT_DIR_PATTERN="./inference_output"

# Analysis script and parameters
ANALYSIS_SCRIPT="analysis/property_analysis_rl.py"
RANDOM_SEED=42

# Default UMAP visualization parameters (will be overridden by stored parameters from inference)
# Leave any parameter as empty string "" to use original defaults from Python script
UMAP_N_NEIGHBORS="20"      # Default: 5 (Higher values (10-50) emphasize global structure)
UMAP_MIN_DIST="1"         # Default: 0.1 (Lower values (0.0-0.3) make tighter clusters)
UMAP_METRIC="jaccard"     # Default: "jaccard" (Distance metric for molecular fingerprints)

# Method mapping - Format is "DisplayName: file_prefix"
# Note: LM and LM-NG are combined since they produce essentially identical results
METHOD_MAPPINGS=(
    "ALM-RL: alm_ppo"
    "ALM: alm"
    "LM-RL: lm_ppo"
    "LM-NG-RL: lm_ng_ppo"
    "LM/LM-NG: lm"
    "LM/LM-NG: lm_ng"
)

# Config file to use
CONFIG_FILE="./config/${BASE_SCAFFOLD}.json"

# Flag to enable/disable initial population comparison
INCLUDE_INITIAL=true
INITIAL_POPULATION_FILE="./training_output/no_scaffold_2_random/2000_initial/initial_population.csv"

#====================================#
# SCRIPT EXECUTION (Don't edit      #
# unless you know what you're doing) #
#====================================#

# Verify the config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    echo "Please check the BASE_SCAFFOLD value and ensure the config file exists."
    exit 1
fi

# Create the main analysis directory
mkdir -p "${ANALYSIS_DIR}"

# Create logs directory
mkdir -p "${LOGS_DIR}"

# Function to create a mapping file for a given scaffold
create_mapping_file() {
    local mutation_folder=$1
    local mapping_file="${ANALYSIS_DIR}/${mutation_folder}/method_mapping.txt"
    
    mkdir -p "${ANALYSIS_DIR}/${mutation_folder}"
    
    # Create mapping file with method display names
    # Format is "DisplayName: file_prefix"
    > "${mapping_file}" # Create empty file
    
    # Write each mapping line to the file
    for mapping in "${METHOD_MAPPINGS[@]}"; do
        echo "$mapping" >> "${mapping_file}"
    done
    
    echo "Created mapping file: ${mapping_file}"
    echo "${mapping_file}"
}

# Prepare initial population flag if enabled
INITIAL_ARGS=""
if [ "$INCLUDE_INITIAL" = true ]; then
    if [ -f "${INITIAL_POPULATION_FILE}" ]; then
        INITIAL_ARGS="--with_initial --initial_molecules ${INITIAL_POPULATION_FILE}"
        echo "Will include initial population from: ${INITIAL_POPULATION_FILE}"
    else
        echo "WARNING: Initial population file not found: ${INITIAL_POPULATION_FILE}"
        echo "Proceeding without initial population comparison."
        INCLUDE_INITIAL=false
    fi
else
    # Even if not including in plots, still pass file for CSV generation if it exists
    if [ -f "${INITIAL_POPULATION_FILE}" ]; then
        INITIAL_ARGS="--initial_molecules ${INITIAL_POPULATION_FILE}"
        echo "Will generate initial population CSV from: ${INITIAL_POPULATION_FILE}"
    fi
fi

# Run analysis for each scaffold variant
for mutation_folder in "${MUTATION_FOLDERS[@]}"; do
    # Create the full scaffold key for this variant
    SCAFFOLD_KEY="${BASE_SCAFFOLD}/${mutation_folder}"
    LOG_FILE="${LOGS_DIR}/${BASE_SCAFFOLD}_${mutation_folder}_analysis.log"
    
    echo "=============================================="
    echo "Running property analysis for ${SCAFFOLD_KEY}"
    echo "Log file: ${LOG_FILE}"
    echo "=============================================="
    
    # Create scaffold-specific output directory
    SCAFFOLD_OUTPUT_DIR="${ANALYSIS_DIR}/${mutation_folder}"
    mkdir -p "${SCAFFOLD_OUTPUT_DIR}"
    
    # Define input directory with generated molecules
    INPUT_DIR="${INPUT_DIR_PATTERN}/${SCAFFOLD_KEY}"
    
    # Check if UMAP parameters file exists from inference step
    UMAP_PARAMS_FILE="${INPUT_DIR}/umap_params.txt"
    if [ -f "${UMAP_PARAMS_FILE}" ]; then
        echo "Loading UMAP parameters from ${UMAP_PARAMS_FILE}"
        # Read the parameters file line by line and extract values
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ $line == UMAP_N_NEIGHBORS=* ]]; then
                PARAM_VALUE="${line#*=}"
                if [ -n "$PARAM_VALUE" ]; then
                    UMAP_N_NEIGHBORS="$PARAM_VALUE"
                fi
            elif [[ $line == UMAP_MIN_DIST=* ]]; then
                PARAM_VALUE="${line#*=}"
                if [ -n "$PARAM_VALUE" ]; then
                    UMAP_MIN_DIST="$PARAM_VALUE"
                fi
            elif [[ $line == UMAP_METRIC=* ]]; then
                PARAM_VALUE="${line#*=}"
                if [ -n "$PARAM_VALUE" ]; then
                    UMAP_METRIC="$PARAM_VALUE"
                fi
            fi
        done < "${UMAP_PARAMS_FILE}"
        echo "Loaded UMAP parameters: neighbors=${UMAP_N_NEIGHBORS}, min_dist=${UMAP_MIN_DIST}, metric=${UMAP_METRIC}"
    else
        echo "No stored UMAP parameters found, using defaults from script"
    fi
    
    # Prepare UMAP parameter flags - always use explicit flags
    UMAP_ARGS="--umap_neighbors ${UMAP_N_NEIGHBORS} --umap_min_dist ${UMAP_MIN_DIST} --umap_metric ${UMAP_METRIC}"
    
    # Pass number of samples to the analysis script
    SAMPLE_ARGS="--num_samples ${NUM_SAMPLES}"
    
    # Start logging
    {
        echo "=============================================="
        echo "Running property analysis for ${SCAFFOLD_KEY}"
        echo "Started at: $(date)"
        echo "Using config file: ${CONFIG_FILE}"
        echo "=============================================="
        echo "Using ${NUM_SAMPLES} samples for error bar calculation and property averaging"
        echo "Using first sample (_1) for UMAP visualization"
        echo "UMAP parameters: neighbors=${UMAP_N_NEIGHBORS}, min_dist=${UMAP_MIN_DIST}, metric=${UMAP_METRIC}"
        
        # Create mapping file for this mutation folder
        MAPPING_FILE=$(create_mapping_file "${mutation_folder}")
        
        # Check if input directory exists
        if [ ! -d "${INPUT_DIR}" ]; then
            echo "Input directory not found for ${SCAFFOLD_KEY}, skipping..."
            echo "Tried looking in: ${INPUT_DIR}"
            continue
        fi
        
        # Check if required files exist and show available files
        echo "Checking available files in ${INPUT_DIR}:"
        ls -la "${INPUT_DIR}" || echo "Cannot list directory contents"
        
        # Run the property analysis script with all parameters
        echo "Processing ${SCAFFOLD_KEY} data and generating plots..."
        
        # Show UMAP parameters being used
        echo "Using UMAP parameters:"
        echo "  n_neighbors: ${UMAP_N_NEIGHBORS}"
        echo "  min_dist: ${UMAP_MIN_DIST}"
        echo "  metric: ${UMAP_METRIC}"
        
        python "${ANALYSIS_SCRIPT}" \
            --best_input_dir "${INPUT_DIR}" \
            --best_output_dir "${SCAFFOLD_OUTPUT_DIR}" \
            --mapping_file "${MAPPING_FILE}" \
            --seed "${RANDOM_SEED}" \
            --config_file "${CONFIG_FILE}" \
            ${UMAP_ARGS} \
            ${SAMPLE_ARGS} \
            ${INITIAL_ARGS}
        
        # Record completion time
        echo "Completed at: $(date)"
        echo "Analysis for ${SCAFFOLD_KEY} completed. Results saved in ${SCAFFOLD_OUTPUT_DIR}"
        echo "----------------------------------------"
    } 2>&1 | tee "${LOG_FILE}"
    
    echo "Analysis for ${SCAFFOLD_KEY} completed. Results saved in ${SCAFFOLD_OUTPUT_DIR}"
    echo "Full log available at: ${LOG_FILE}"
    echo "----------------------------------------"
done

echo "All property analyses completed!"
echo "Results are organized in mutation-specific folders under ${ANALYSIS_DIR}"
echo "All logs are available in ${LOGS_DIR}" 