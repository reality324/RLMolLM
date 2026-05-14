"""
手性分子处理工具

提供手性检测和转换功能，用于处理手性分子输入。
"""
from rdkit import Chem


def has_chirality(smiles: str) -> bool:
    """
    检测分子是否有手性。
    
    Args:
        smiles: 分子的 SMILES 字符串
        
    Returns:
        bool: 如果分子有手性中心则返回 True
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            return True
    return False


def get_chiral_info(smiles: str) -> dict:
    """
    获取分子的手性详细信息。
    
    Args:
        smiles: 分子的 SMILES 字符串
        
    Returns:
        dict: 包含手性信息的字典
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'valid': False, 'chiral_atoms': [], 'chiral_bonds': []}
    
    chiral_atoms = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            chiral_atoms.append({
                'symbol': atom.GetSymbol(),
                'cip_code': atom.GetProp('_CIPCode'),
                'idx': atom.GetIdx()
            })
    
    chiral_bonds = []
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            chiral_bonds.append({
                'atoms': [a.GetSymbol() for a in bond.GetAtoms()],
                'stereo': str(bond.GetStereo())
            })
    
    return {
        'valid': True,
        'has_chirality': len(chiral_atoms) > 0 or len(chiral_bonds) > 0,
        'chiral_atoms': chiral_atoms,
        'chiral_bonds': chiral_bonds
    }


def remove_chirality(smiles: str) -> str:
    """
    将手性分子转换为非手性版本。
    移除所有立体化学信息（@, @@ 等）。
    
    Args:
        smiles: 分子的 SMILES 字符串
        
    Returns:
        str: 非手性版本的 SMILES 字符串，如果输入无效则返回 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def convert_to_nonchiral_input(smiles: str, verbose: bool = True) -> dict:
    """
    智能转换输入分子：
    - 如果无手性，直接返回
    - 如果有手性，转换成非手性版本
    
    Args:
        smiles: 输入的 SMILES 字符串
        verbose: 是否打印转换信息
        
    Returns:
        dict: 包含以下键的字典：
            - original: 原始 SMILES
            - converted: 转换后的 SMILES
            - was_chiral: 是否原本有手性
            - chiral_info: 手性详细信息
            - valid: SMILES 是否有效
    """
    result = {
        'original': smiles,
        'converted': None,
        'was_chiral': False,
        'chiral_info': None,
        'valid': False
    }
    
    # 检查有效性
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"警告: 无法解析 SMILES: {smiles}")
        return result
    
    result['valid'] = True
    
    # 获取手性信息
    result['chiral_info'] = get_chiral_info(smiles)
    result['was_chiral'] = result['chiral_info']['has_chirality']
    
    if result['was_chiral']:
        if verbose:
            chiral_atoms = result['chiral_info']['chiral_atoms']
            print(f"检测到手性分子，手性中心: {len(chiral_atoms)} 个")
            for ca in chiral_atoms[:3]:  # 只显示前3个
                print(f"  - {ca['symbol']} (位置 {ca['idx']}): {ca['cip_code']}")
            if len(chiral_atoms) > 3:
                print(f"  ... 还有 {len(chiral_atoms) - 3} 个")
            print(f"转换为非手性版本继续优化...")
        
        result['converted'] = remove_chirality(smiles)
    else:
        result['converted'] = smiles
    
    return result


def smart_convert_input(smiles_or_list, verbose: bool = True) -> dict:
    """
    智能转换输入（支持单个分子或列表）。
    
    Args:
        smiles_or_list: 单个 SMILES 或 SMILES 列表
        verbose: 是否打印信息
        
    Returns:
        dict: 转换结果
    """
    if isinstance(smiles_or_list, (list, tuple)):
        # 处理列表
        results = []
        for smi in smiles_or_list:
            results.append(convert_to_nonchiral_input(smi, verbose=verbose))
        return {'type': 'list', 'results': results}
    else:
        # 处理单个
        return {'type': 'single', 'result': convert_to_nonchiral_input(smiles_or_list, verbose=verbose)}
