import re

def smiles_to_char_list(smiles_string):
    """
    Convert a SMILES string to a list of characters/tokens based on the same regex pattern 
    used by the tokenizer in the codebase.
    
    Args:
        smiles_string (str): A SMILES string (e.g., 'c1ccccn1')
        
    Returns:
        list: A list of characters/tokens from the SMILES string
    """
    # This regex pattern matches the same pattern used in the tokenizer
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    
    # Use findall to extract all tokens that match the pattern
    tokens = re.findall(pattern, smiles_string)
    
    # Remove any empty strings that might appear in the result
    tokens = [token for token in tokens if token]
    
    return tokens

# Example usage
if __name__ == "__main__":
    # Example SMILES string
    smiles = "c1ccccn1"
    
    # Convert to character list
    char_list = smiles_to_char_list(smiles)
    
    # Print the result
    print(f"SMILES: {smiles}")
    print(f"Character list: {char_list}")
    
    # Show the example you provided
    print("\nExample from user:")
    print(f"SMILES: c1ccccn1")
    print(f"Character list: {smiles_to_char_list('c1ccccn1')}")
    
    # Additional example
    print("\nAnother example:")
    print(f"SMILES: CC(=O)C")
    print(f"Character list: {smiles_to_char_list('CC(=O)C')}") 