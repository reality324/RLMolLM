# RLMolLM: Reinforcement Learning-Enhanced Language Model Framework for Inverse Molecular Design

This repository contains code for training and evaluating molecular generation models using reinforcement learning approaches. The framework supports both scaffold-based and non-scaffold molecular generation with various training configurations.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow Overview](#workflow-overview)
- [Step 1: Generate Initial Population](#step-1-generate-initial-population)
- [Step 2: Training](#step-2-training)
- [Step 3: Inference](#step-3-inference)
- [Step 4: Analysis](#step-4-analysis)
- [Configuration](#configuration)
- [Output Structure](#output-structure)

## Installation

### 1. Create Conda Environment

```bash
# Create the Python 3.11 environment
conda create -n rlmollm-env python=3.11.11
conda activate rlmollm-env
```

### 2. Install Dependencies

```bash
# Install the requirements
pip install -r requirements.txt
```

### 3. Download Required Model Files

Due to file size limitations, some large files are hosted separately:

**Model Weights (209MB):**
- Download from: [To be uploaded to Hugging Face Hub]
- Extract to: `model_weights/`



**Note**: This project requires Python 3.11.11 and specific package versions as listed in `requirements.txt`. The main dependencies include:
- PyTorch 2.4.0
- Transformers 4.50.0
- RDKit 2024.3.3
- ADMET-AI 1.3.1 (ADMET features in development)
- UMAP-learn 0.5.7

## Project Structure

```
├── config/                     # Configuration files
│   ├── scaffold_examples/      # Scaffold-specific configs
│   └── no_scaffold_2_random.json  # Non-scaffold config
├── models/                     # Neural network model definitions
│   ├── rl/                     # Reinforcement learning models
│   ├── generator.py            # Generator model architecture
│   ├── discriminator.py        # Discriminator model architecture (not used)
│   └── gan.py                  # GAN implementation (only generator used)
├── analysis/                   # Analysis scripts and outputs
│   ├── property_analysis_rl.py # Main analysis script
│   ├── umap_plot.py           # UMAP visualization
│   ├── plots/                 # Generated plots and visualizations
│   └── logs/                  # Analysis logs
├── scoring/                    # Molecular scoring functions
│   ├── molecule_scoring.py     # Core scoring functions
│   ├── admet_scoring.py       # Molecular scoring with ADMET features in development
│   └── scoring_interface.py   # Scoring interface
├── utils/                      # Utility functions
│   ├── util.py                # Core utilities
│   ├── training_utils.py      # Training helper functions
│   └── sample_util.py         # Sampling utilities
├── population/                 # Population management
├── model_weights/             # Pre-trained model weights
├── scaffold/                  # Scaffold-related functionality
├── tokenizer/                 # Molecular tokenization
├── data/                      # Data storage
├── training_output/           # Training results and checkpoints
├── inference_output/          # Generated molecules from inference
├── scaffold_database/         # Scaffold database files
├── api/                       # Web API components (optional)
├── build_initial_population_*.sh  # Initial population generation
├── train_all_*.sh            # Training scripts
├── run_inference_*.sh        # Inference scripts
├── run_property_analysis_*.sh # Analysis scripts
├── training_combined.py       # Main training script
├── inference.py              # Main inference script
└── requirements.txt          # Python dependencies
```

## Workflow Overview

The complete workflow consists of 4 main steps:

1. **Generate Initial Population** - Create starting molecules for training
2. **Training** - Train molecular generation models
3. **Inference** - Generate new molecules using trained models
4. **Analysis** - Analyze and visualize results

Each step has two variants:
- **Non-scaffold (`_ns_random`)**: Generates molecules without scaffold constraints (unconstrained)
- **Scaffold-based (`_s`)**: Uses molecular scaffolds as constraints

### Script Configuration

All shell scripts contain a **"CONFIGURABLE VARIABLES (Edit here)"** section where you can customize:

- **Model Selection**: Choose which training methods to use via `CONFIG_KEYS` array
- **Training Parameters**: Epochs, population size, mutation rates, etc.
- **Output Directories**: Customize where results are saved
- **Scaffold Selection**: For scaffold-based workflows

**Available Model Types**:
```bash
CONFIG_KEYS=(
    "alm_ppo"    # Adaptive Language Model (adaptive genetic algo) with PPO RL
    "alm"        # Adaptive Language Model (adaptive genetic algo)
    "lm_ppo"     # Language Model with PPO RL and genetic algo
    "lm_ng_ppo"  # Language Model with PPO RL but no genetic algo
    "lm"         # Language Model with genetic algo (no weight updates)
    "lm_ng"      # Language Model with no genetic algo (no weight updates)
)
```

**Note**: `lm`/`lm_ng` models have no weight updates during training - only maintained elite population differs.

## Step 1: Generate Initial Population

Before training, you need to generate an initial population of molecules or use your own dataset.

### For Non-scaffold Generation (Unconstrained):
```bash
./build_initial_population_ns.sh
```

### For Scaffold-based Generation (Constrained):
```bash
./build_initial_population_s.sh
```

**Important**: 
- These scripts automatically create a `2000_initial/` directory structure and save the generated `initial_population.csv` file there
- The output will be organized as:
  - **Non-scaffold**: `training_output/no_scaffold_2_random/2000_initial/initial_population.csv`
  - **Scaffold-based**: `training_output/{scaffold_name}/2000_initial/initial_population.csv`

**Customization**: 
- Edit the scripts to modify population size, output directories, or scaffold configurations
- You can also provide your own initial population CSV file with SMILES molecules in the same directory structure

## Step 2: Training

Train molecular generation models with various configurations.

### For Non-scaffold Training (Unconstrained):
```bash
./train_all_ns_random.sh
```

This script trains models with different mutation parameters (1.0, 0.8, 0.7, 0.6, 0.5) and configurable training methods. You can select which models to train by editing the `CONFIG_KEYS` array in the script (same options as scaffold-based training above).

**Customization**: Edit the configurable variables in the "CONFIGURABLE VARIABLES (Edit here)" section, including:
- `CONFIG_KEYS`: Select which training methods to use
- `MUTATION_RATES`: Adjust mutation parameters for exploration vs exploitation
- Other training parameters like epochs, population size, etc.

### For Scaffold-based Training (Constrained):
```bash
./train_all_s.sh
```

This script trains models for multiple scaffolds with different methods. You can configure which models to train by editing the `CONFIG_KEYS` array and other parameters in the "CONFIGURABLE VARIABLES (Edit here)" section of the script.


## Step 3: Inference

Generate new molecules using the trained models.

### For Non-scaffold Inference (Unconstrained):
```bash
./run_inference_ns_random.sh
```

### For Scaffold-based Inference (Constrained):
```bash
./run_inference_s.sh
```

**Inference Parameters**:
- Multiple validation modes:
  - `valid_unique_only`: Only valid and unique molecules
  - `valid_only`: Only valid molecules
  - `no_validation`: Any generated molecules
- Multiple repetitions for statistical analysis (4-8 samples per method)

## Step 4: Analysis

Analyze the generated molecules and create visualizations.

### For Non-scaffold Analysis (Unconstrained):
```bash
./run_property_analysis_ns_random.sh
```

### For Scaffold-based Analysis (Constrained):
```bash
./run_property_analysis_s.sh
```

**Analysis Outputs**:
- **Property distributions**: LogP, molecular weight, TPSA, etc.
- **UMAP visualizations**: 2D projections of molecular chemical space
- **Validity and uniqueness statistics**: Error bars across multiple runs
- **Comparison plots**: Different methods and initial populations

## Configuration

### Non-scaffold Configuration (Unconstrained)
- Configuration file: `config/no_scaffold_2_random.json`
- Supports different mutation parameters for exploration vs exploitation

### Scaffold Configuration (Constrained)
- Available scaffolds: `scaffold_6_benzene`, `scaffold_7_dihydropyridine`, `scaffold_8_benzothiophene`
- Configuration files in `config/scaffold_examples/`

## Output Structure

### Training Output
```
training_output/
├── scaffold_6_benzene/
│   ├── alm_ppo_2000_t1_e20/    # Trained models
│   ├── alm_2000_t1_e20/
│   └── ...
└── no_scaffold_2_random/
    ├── 1m/                      # Mutation parameter 1.0
    │   ├── alm_ppo/
    │   └── ...
    └── 0p7m/                    # Mutation parameter 0.7
```

### Inference Output
```
inference_output/
├── scaffold_6_benzene/
│   ├── alm_ppo_valid_unique_only_1.csv
│   ├── alm_ppo_valid_only_1.csv
│   └── ...
└── no_scaffold_2_random/
    └── 1m/
        ├── alm_ppo_valid_unique_only_1.csv
        └── ...
```

## Technical Notes

**No GAN Model Used**: While the codebase includes GAN components (generator and discriminator), no GAN training is performed due to mode collapse issues commonly encountered with GANs in molecular generation. The framework focuses on language model approaches with optional reinforcement learning enhancement.
