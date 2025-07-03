#!/bin/bash

#====================================#
# CONFIGURABLE VARIABLES (Edit here) #
#====================================#

# Scaffold configurations to analyze
SCAFFOLD_KEYS=(
    "scaffold_6_benzene"
    # "scaffold_7_dihydropyridine"
    # "scaffold_8_benzothiophene"
)

# Number of samples per method (for error bar calculation)
# Note: Only the first sample (_1) will be used for property curves and UMAP visualization
# All samples will be used to calculate error bars for validity and uniqueness
NUM_SAMPLES=4

# Base directories
ANALYSIS_DIR="./analysis/plots"
LOGS_DIR="./analysis/logs"

# Input directory pattern (where generated molecules are stored)
INPUT_DIR_PATTERN="./inference_output"

# Analysis script and parameters
ANALYSIS_SCRIPT="analysis/property_analysis_rl.py"
RANDOM_SEED=42

# UMAP visualization parameters
# Leave any parameter as empty string "" to use original defaults from Python script
UMAP_N_NEIGHBORS="5"     # Default: 5 (Higher values (10-50) emphasize global structure)
UMAP_MIN_DIST="1"       # Default: 0.1 (Lower values (0.0-0.3) make tighter clusters)
UMAP_METRIC=""            # Default: "jaccard" (Distance metric for molecular fingerprints)

# Method mapping - Format is "DisplayName: file_prefix"
# Note: LM and LM-NG are combined into "LM/LM-NG" in the analysis script
# since they produce identical results (neither actually trains model weights)
METHOD_MAPPINGS=(
    "ALM-RL: alm_ppo"
    "ALM: alm"
    "LM-RL: lm_ppo"
    "LM-NG-RL: lm_ng_ppo"
    "LM/LM-NG: lm"
    "LM/LM-NG: lm_ng"
)

# Flag to enable/disable initial population comparison
INCLUDE_INITIAL=false

# Initial population files mapping - Format: "scaffold_key: path_to_initial_population_file"
# Each scaffold can have its own initial population file
declare -A INITIAL_POPULATION_FILES
INITIAL_POPULATION_FILES["scaffold_1_acry"]="./training_output/scaffold_1_acry/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_2_indole"]="./training_output/scaffold_2_indole/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_3_biphe"]="./training_output/scaffold_3_biphe/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_4"]="./training_output/scaffold_4/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_5_male"]="./training_output/scaffold_5_male/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_6_benzene"]="./training_output/scaffold_6_benzene/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_7_dihydropyridine"]="./training_output/scaffold_7_dihydropyridine/2000_initial/initial_population.csv"
INITIAL_POPULATION_FILES["scaffold_8_benzothiophene"]="./training_output/scaffold_8_benzothiophene/2000_initial/initial_population.csv"

#====================================#
# SCRIPT EXECUTION (Don't edit      #
# unless you know what you're doing) #
#====================================#

# Create the main analysis directory
mkdir -p "${ANALYSIS_DIR}"

# Create logs directory
mkdir -p "${LOGS_DIR}"

# Function to create a mapping file for a given scaffold
create_mapping_file() {
    local scaffold=$1
    local mapping_file="${ANALYSIS_DIR}/${scaffold}/method_mapping.txt"
    
    mkdir -p "${ANALYSIS_DIR}/${scaffold}"
    
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

# Run analysis for each scaffold
for scaffold_key in "${SCAFFOLD_KEYS[@]}"; do
    LOG_FILE="${LOGS_DIR}/${scaffold_key}_analysis.log"
    
    echo "=============================================="
    echo "Running property analysis for ${scaffold_key}"
    echo "Log file: ${LOG_FILE}"
    echo "=============================================="
    
    # Create scaffold-specific output directory
    SCAFFOLD_OUTPUT_DIR="${ANALYSIS_DIR}/${scaffold_key}"
    mkdir -p "${SCAFFOLD_OUTPUT_DIR}"
    
    # Prepare scaffold-specific initial population arguments
    SCAFFOLD_INITIAL_ARGS=""
if [ "$INCLUDE_INITIAL" = true ]; then
        # Get the initial population file for this specific scaffold
        SCAFFOLD_INITIAL_FILE="${INITIAL_POPULATION_FILES[$scaffold_key]}"
        if [ -n "$SCAFFOLD_INITIAL_FILE" ] && [ -f "$SCAFFOLD_INITIAL_FILE" ]; then
            SCAFFOLD_INITIAL_ARGS="--with_initial --initial_molecules $SCAFFOLD_INITIAL_FILE"
            echo "Will include initial population from: $SCAFFOLD_INITIAL_FILE"
    else
            echo "WARNING: Initial population file not found for ${scaffold_key}: $SCAFFOLD_INITIAL_FILE"
            echo "Proceeding without initial population comparison for ${scaffold_key}."
        fi
    else
        # Even if not including in plots, still pass file for CSV generation if it exists
        SCAFFOLD_INITIAL_FILE="${INITIAL_POPULATION_FILES[$scaffold_key]}"
        if [ -n "$SCAFFOLD_INITIAL_FILE" ] && [ -f "$SCAFFOLD_INITIAL_FILE" ]; then
            SCAFFOLD_INITIAL_ARGS="--initial_molecules $SCAFFOLD_INITIAL_FILE"
            echo "Will generate initial population CSV for ${scaffold_key} from: $SCAFFOLD_INITIAL_FILE"
    fi
fi

# Prepare UMAP parameter flags
UMAP_ARGS=""
if [ -n "$UMAP_N_NEIGHBORS" ]; then
    UMAP_ARGS="$UMAP_ARGS --umap_neighbors $UMAP_N_NEIGHBORS"
fi
if [ -n "$UMAP_MIN_DIST" ]; then
    UMAP_ARGS="$UMAP_ARGS --umap_min_dist $UMAP_MIN_DIST"
fi
if [ -n "$UMAP_METRIC" ]; then
    UMAP_ARGS="$UMAP_ARGS --umap_metric $UMAP_METRIC"
fi

# Pass number of samples to the analysis script
SAMPLE_ARGS="--num_samples ${NUM_SAMPLES}"
    
    # Start logging
    {
        echo "=============================================="
        echo "Running property analysis for ${scaffold_key}"
        echo "Started at: $(date)"
        echo "=============================================="
        echo "Using ${NUM_SAMPLES} samples for error bar calculation"
        echo "Using first sample (_1) for property curves and UMAP visualization"
        
        # Create mapping file for this scaffold
        MAPPING_FILE=$(create_mapping_file "${scaffold_key}")
        
        # Define input directory with generated molecules
        INPUT_DIR="${INPUT_DIR_PATTERN}/${scaffold_key}"
        
        # Check if input directory exists
        if [ ! -d "${INPUT_DIR}" ]; then
            echo "Input directory not found for ${scaffold_key}, skipping..."
            echo "Tried looking in: ${INPUT_DIR}"
            continue
        fi
        
        # Check if required files exist and show available files
        echo "Checking available files in ${INPUT_DIR}:"
        ls -la "${INPUT_DIR}" || echo "Cannot list directory contents"
        
        # Determine config file location based on scaffold name
        if [[ "${scaffold_key}" == no_scaffold* ]]; then
            # Handle no_scaffold and variants like no_scaffold_0p8m
            if [ "${scaffold_key}" == "no_scaffold" ]; then
                CONFIG_FILE="./config/no_scaffold.json"
            else
                # First try the direct variant config
                CONFIG_FILE="./config/${scaffold_key}.json"
                
                # If not found, try in no_scaffold subdirectory
                if [ ! -f "${CONFIG_FILE}" ]; then
                    CONFIG_FILE="./config/no_scaffold/${scaffold_key}.json"
                fi
                
                # Fallback to the standard no_scaffold config if the variant config doesn't exist
                if [ ! -f "${CONFIG_FILE}" ]; then
                    CONFIG_FILE="./config/no_scaffold.json"
                    echo "Variant config not found, falling back to: ${CONFIG_FILE}"
                fi
            fi
        else
            CONFIG_FILE="./config/scaffold_examples/${scaffold_key}.json"
        fi
        
        # Verify config file exists and print info
        CONFIG_ARGS=""
        if [ -f "${CONFIG_FILE}" ]; then
            echo "Found config: ${CONFIG_FILE}"
            echo "Using config file: ${CONFIG_FILE}"
            
            # Extract logP preferred range for debugging
            echo "=== Property Config Section ==="
            grep -A 10 "preferred_range" "${CONFIG_FILE}" || echo "No preferred_range found in config"
            echo "==========================="
            
            # Pass config file to Python script
            CONFIG_ARGS="--config_file ${CONFIG_FILE}"
        else
            echo "Warning: Config not found at ${CONFIG_FILE}"
            echo "Will use default logP vertical lines (2 and 3)"
        fi
        
        # Show UMAP parameters being used (if any)
        if [ -n "$UMAP_ARGS" ]; then
            echo "Using custom UMAP parameters:"
            [ -n "$UMAP_N_NEIGHBORS" ] && echo "  n_neighbors: $UMAP_N_NEIGHBORS"
            [ -n "$UMAP_MIN_DIST" ] && echo "  min_dist: $UMAP_MIN_DIST"
            [ -n "$UMAP_METRIC" ] && echo "  metric: $UMAP_METRIC"
        else
            echo "Using default UMAP parameters from Python script"
        fi
        
        # Run the property analysis script with all parameters
        echo "Processing ${scaffold_key} data and generating plots..."
        python "${ANALYSIS_SCRIPT}" \
            --best_input_dir "${INPUT_DIR}" \
            --best_output_dir "${SCAFFOLD_OUTPUT_DIR}" \
            --mapping_file "${MAPPING_FILE}" \
            --seed "${RANDOM_SEED}" \
            ${CONFIG_ARGS} \
            ${UMAP_ARGS} \
            ${SAMPLE_ARGS} \
            ${SCAFFOLD_INITIAL_ARGS}
        
        # Record completion time
        echo "Completed at: $(date)"
        echo "Analysis for ${scaffold_key} completed. Results saved in ${SCAFFOLD_OUTPUT_DIR}"
        echo "----------------------------------------"
    } 2>&1 | tee "${LOG_FILE}"
    
    echo "Analysis for ${scaffold_key} completed. Results saved in ${SCAFFOLD_OUTPUT_DIR}"
    echo "Full log available at: ${LOG_FILE}"
    echo "----------------------------------------"
done

echo "All property analyses completed!"
echo "Results are organized in scaffold-specific folders under ${ANALYSIS_DIR}"
echo "All logs are available in ${LOGS_DIR}" 