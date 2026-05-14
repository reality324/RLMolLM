"""
All available ADMET property configurations.
Extracted from admet-ai package and literature.

Each property has:
- range: [min, max] for normalization
- preferred_value: target value for optimization (closer is better)
  - If higher is better: use max value
  - If lower is better: use min value  
  - If specific target: use that value
"""

# All available ADMET and molecular properties with their configurations
PROPERTY_CONFIGS = {

    # === RDKit properties ===
    "synth": {"range": [0, 8], "preferred_value": 0.0},
    "drug": {"range": [0, 1], "preferred_value": 1.0},
    "logP": {"range": [-4, 7], "preferred_range": [1.35, 1.8]},
    "logD": {"range": [-4, 7], "preferred_range": [1.35, 1.8]},

    # === Absorption/Permeability ===
    "Pgp_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "Pgp_inhibitor": {"range": [0, 1], "preferred_value": 1.0},
    "BCRP_inhibitor": {"range": [0, 1], "preferred_value": 1.0},
    "BSEP_inhibitor": {"range": [0, 1], "preferred_value": 1.0},
    "HIA": {"range": [0, 1], "preferred_value": 1.0},
    "PAMPA": {"range": [0, 1], "preferred_value": 1.0},
    "F20%": {"range": [0, 1], "preferred_value": 1.0},
    "F30%": {"range": [0, 1], "preferred_value": 1.0},
    "F50%": {"range": [0, 1], "preferred_value": 1.0},
    "OATP1B1_inhibitor": {"range": [0, 1], "preferred_value": 1.0},
    "OATP1B3_inhibitor": {"range": [0, 1], "preferred_value": 1.0},
    "BBB": {"range": [0, 1], "preferred_value": 1.0},
    "MRP1_inhibitor": {"range": [0, 1], "preferred_value": 1.0},

    # === Metabolism/CYP ===
    "CYP3A4_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP1A2_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP1A2_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP2C19_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2C19_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP2C9_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2C9_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP2D6_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2D6_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP3A4_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP2B6_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2B6_substrate": {"range": [0, 1], "preferred_value": 1.0},
    "CYP2C8_inhibitor": {"range": [0, 1], "preferred_value": 0.0},
    "HLM_stability": {"range": [0, 1], "preferred_value": 1.0},

    # === Toxicity (LiTEN) ===
    "hERG_Blockers": {"range": [0, 1], "preferred_value": 0.0},
    "hERG_Blockers_10um": {"range": [0, 1], "preferred_value": 0.0},
    "AMES_Mutagenicity": {"range": [0, 1], "preferred_value": 0.0},
    "Drug_induced_liver_injury": {"range": [0, 1], "preferred_value": 1.0},
    "ROA": {"range": [0, 1], "preferred_value": 1.0},
    "FDAMDD": {"range": [0, 1], "preferred_value": 1.0},
    "Skin_Sensitization": {"range": [0, 1], "preferred_value": 1.0},
    "Carcinogenicity": {"range": [0, 1], "preferred_value": 1.0},
    "Eye_Corrosion": {"range": [0, 1], "preferred_value": 1.0},
    "Eye_Irritation": {"range": [0, 1], "preferred_value": 1.0},
    "Respiratory": {"range": [0, 1], "preferred_value": 1.0},
    "Human_Hepatotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Drug_induced_Neurotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Ototoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Hematotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Drug_induced_Nephrotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Genotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "RPMI_8226_Immunitoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "A549_Cytotoxicity": {"range": [0, 1], "preferred_value": 1.0},
    "Hek293_Cytotoxicity": {"range": [0, 1], "preferred_value": 1.0},

    # === Tox21 NR ===
    "NR_AhR": {"range": [0, 1], "preferred_value": 1.0},
    "NR_AR": {"range": [0, 1], "preferred_value": 1.0},
    "NR_AR_LBD": {"range": [0, 1], "preferred_value": 1.0},
    "NR_Aromatase": {"range": [0, 1], "preferred_value": 1.0},
    "NR_ER": {"range": [0, 1], "preferred_value": 1.0},
    "NR_ER_LBD": {"range": [0, 1], "preferred_value": 1.0},
    "NR_PPAR_gamma": {"range": [0, 1], "preferred_value": 1.0},

    # === Tox21 SR ===
    "SR_ARE": {"range": [0, 1], "preferred_value": 1.0},
    "SR_ATAD5": {"range": [0, 1], "preferred_value": 1.0},
    "SR_HSE": {"range": [0, 1], "preferred_value": 1.0},
    "SR_MMP": {"range": [0, 1], "preferred_value": 1.0},
    "SR_p53": {"range": [0, 1], "preferred_value": 1.0},

    # === Additional reference properties ===
    "AMES": {"range": [0, 1], "preferred_value": 0.0},
    "BBB_Martins": {"range": [0, 1], "preferred_value": 0.0},
    "BCF": {"range": [-5, 10], "preferred_value": 0.0},
    "Bioavailability_Ma": {"range": [0, 1], "preferred_value": 1.0},
    "CYP1A2_Veith": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2C19_Veith": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2C9_Substrate_CarbonMangels": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2C9_Veith": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2D6_Substrate_CarbonMangels": {"range": [0, 1], "preferred_value": 0.0},
    "CYP2D6_Veith": {"range": [0, 1], "preferred_value": 0.0},
    "CYP3A4_Substrate_CarbonMangels": {"range": [0, 1], "preferred_value": 0.0},
    "CYP3A4_Veith": {"range": [0, 1], "preferred_value": 0.0},
    "Caco2_Wang": {"range": [-8, 2], "preferred_value": 2.0},
    "Carcinogens_Lagunin": {"range": [0, 1], "preferred_value": 0.0},
    "Cl_Plasma": {"range": [-10, 10], "preferred_value": 0.0},
    "Clearance_Hepatocyte_AZ": {"range": [-10, 10], "preferred_value": 0.0},
    "Clearance_Microsome_AZ": {"range": [-10, 10], "preferred_value": 0.0},
    "ClinTox": {"range": [0, 1], "preferred_value": 0.0},
    "DILI": {"range": [0, 1], "preferred_value": 0.0},
    "HIA_Hou": {"range": [0, 1], "preferred_value": 1.0},
    "Half_Life_Obach": {"range": [-30, 30], "preferred_value": 0.0},
    "HydrationFreeEnergy_FreeSolv": {"range": [-30, 10], "preferred_value": -7.5},
    "IGC50": {"range": [-5, 10], "preferred_value": 8.0},
    "LC50DM": {"range": [-5, 10], "preferred_value": 8.0},
    "LC50FM": {"range": [-5, 10], "preferred_value": 8.0},
    "LD50_Zhu": {"range": [0, 5], "preferred_value": 5.0},
    "Lipinski": {"range": [0, 4], "preferred_value": 0.0},
    "Lipophilicity_AstraZeneca": {"range": [-3, 3], "preferred_value": 0.5},
    "PAMPA_NCATS": {"range": [0, 1], "preferred_value": 1.0},
    "PPBR_AZ": {"range": [0, 100], "preferred_value": 50.0},
    "Pgp_Broccatelli": {"range": [0, 1], "preferred_value": 0.0},
    "QED": {"range": [0, 1], "preferred_value": 1.0},
    "Skin_Reaction": {"range": [0, 1], "preferred_value": 0.0},
    "Solubility_AqSolDB": {"range": [-10, 2], "preferred_value": -1.0},
    "VDss_Lombardo": {"range": [-5, 5], "preferred_value": 0.0},
    "hERG": {"range": [0, 1], "preferred_value": 0.0},
    "hydrogen_bond_acceptors": {"range": [0, 15], "preferred_value": 5.0},
    "hydrogen_bond_donors": {"range": [0, 8], "preferred_value": 2.5},
    "molecular_weight": {"range": [0, 900], "preferred_value": 300.0},
    "stereo_centers": {"range": [0, 10], "preferred_value": 1.5},
    "tpsa": {"range": [0, 200], "preferred_value": 80.0},
}

def get_property_config(property_name):
    """
    Get configuration for a specific ADMET/molecular property.
    
    Args:
        property_name: Name of the property (e.g., 'PAMPA_NCATS', 'hERG')
        
    Returns:
        Dict with 'range' and 'preferred_value' keys, or None if not found
    """
    return PROPERTY_CONFIGS.get(property_name)


def merge_property_configs(user_config, additional_properties):
    """
    Merge user's property_config with configurations from PROPERTY_CONFIGS
    for additional properties.
    
    Args:
        user_config: User's existing property_config dict
        additional_properties: List of property names to add from PROPERTY_CONFIGS
        
    Returns:
        Merged property_config dict
    """
    merged_config = user_config.copy()
    
    for prop_name in additional_properties:
        if prop_name not in merged_config and prop_name in PROPERTY_CONFIGS:
            merged_config[prop_name] = PROPERTY_CONFIGS[prop_name].copy()
    
    return merged_config


def get_all_properties():
    """Get list of all available property names."""
    return list(PROPERTY_CONFIGS.keys())


if __name__ == "__main__":
    # Quick test
    print(f"Total properties available: {len(PROPERTY_CONFIGS)}")
    print(f"\nSample - PAMPA_NCATS: {get_property_config('PAMPA_NCATS')}")
    print(f"Sample - hERG: {get_property_config('hERG')}")
    print(f"Sample - logP: {get_property_config('logP')}")
    
    # Test merge
    user_config = {
        'synth': {'range': [0, 8], 'higher_is_better': False},
        'drug': {'range': [0, 1], 'higher_is_better': True}
    }
    merged = merge_property_configs(user_config, ['PAMPA_NCATS', 'AMES'])
    print(f"\nMerged config has {len(merged)} properties")
    print(f"Added: {[k for k in merged.keys() if k not in user_config]}")
