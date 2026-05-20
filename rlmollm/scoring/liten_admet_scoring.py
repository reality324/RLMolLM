import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class LiTENModelSpec:
    task_class: str
    ckpt_path: str
    is_reg: bool
    task_names: List[str]


def _get_num_tasks_from_ckpt(ckpt_dict) -> int:
    if isinstance(ckpt_dict, dict) and 'num_tasks' in ckpt_dict:
        return int(ckpt_dict['num_tasks'])

    state_dict = ckpt_dict.get('model_state_dict', ckpt_dict)
    max_idx = -1
    for key in state_dict.keys():
        match = re.search(r'heads\.(\d+)\.', key)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    if max_idx >= 0:
        return max_idx + 1
    raise ValueError("Unable to infer num_tasks from checkpoint.")


def _mol_to_pyg_data(mol, uid: str) -> Data:
    if mol.GetNumAtoms() == 0:
        return None
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    z = torch.tensor(atomic_nums, dtype=torch.long)
    try:
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    except Exception:
        return None
    data = Data(z=z, pos=pos)
    data.uid = [uid]
    return data


def _generate_lowest_energy_conformers(smiles: str, num_confs: int, top_k: int) -> List:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(result) == 0:
        return []
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, nonBondedThresh=100.0)
    except Exception:
        return []
    energies: List[Tuple[int, float]] = []
    for idx, (not_converged, energy) in enumerate(res):
        if not_converged == 0:
            energies.append((idx, float(energy)))
    energies.sort(key=lambda x: x[1])
    top_indices = [idx for idx, _ in energies[:top_k]]
    if not top_indices:
        return []
    kept: List = []
    for idx in top_indices:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(mol.GetConformer(idx), assignId=True)
        kept.append(new_mol)
    return kept


class LiTENADMETPredictor:
    def __init__(
        self,
        base_path: str,
        ckpt_dir: str,
        conf_json: str,
        device: Optional[str] = None,
        num_confs: int = 3,
        top_k_confs: int = 1,
        batch_size: int = 32,
        cache_size: int = 20000,
    ):
        self.base_path = base_path
        self.ckpt_dir = ckpt_dir
        self.conf_json = conf_json
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_confs = int(num_confs)
        self.top_k_confs = int(top_k_confs)
        self.batch_size = int(batch_size)

        with open(conf_json, 'r') as f:
            self._task_dict = json.load(f)

        self._models: Dict[str, Tuple[torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor], LiTENModelSpec]] = {}

        self._cache_size = int(cache_size)
        self._cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._cache_order: List[Tuple[str, str]] = []

    def _cache_get(self, task_class: str, smiles: str) -> Optional[Dict[str, float]]:
        key = (task_class, smiles)
        return self._cache.get(key)

    def _cache_set(self, task_class: str, smiles: str, values: Dict[str, float]) -> None:
        key = (task_class, smiles)
        if key in self._cache:
            self._cache[key] = values
            return
        self._cache[key] = values
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def _load_task(self, task_class: str):
        if task_class in self._models:
            return self._models[task_class]

        ckpt_path = os.path.join(self.ckpt_dir, f"{task_class}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"LiTEN checkpoint not found: {ckpt_path}")

        is_reg = task_class.endswith('_reg')
        task_names = self._task_dict.get(task_class)
        
        torch.serialization.add_safe_globals(['model.LiTEN_ADMETLAB.LiTEN'])
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        num_tasks = _get_num_tasks_from_ckpt(ckpt)

        if task_names is None or len(task_names) != num_tasks:
            task_names = [f"Task_{i + 1}" for i in range(num_tasks)]

        abs_path = os.path.abspath(self.base_path)
        if abs_path not in os.sys.path:
            os.sys.path.append(abs_path)

        from model.LiTEN_ADMETLAB import LiTEN  # type: ignore

        model = LiTEN(num_tasks=num_tasks, task_class='reg' if is_reg else 'cls').to(self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)

        mean_t, std_t = None, None
        if is_reg:
            if isinstance(ckpt, dict) and 'mean' in ckpt and 'std' in ckpt:
                mean_t = ckpt['mean'].to(self.device)
                std_t = ckpt['std'].to(self.device)
            else:
                # Checkpoint doesn't have mean/std - use identity normalization
                num_tasks = _get_num_tasks_from_ckpt(ckpt)
                mean_t = torch.zeros(num_tasks, device=self.device)
                std_t = torch.ones(num_tasks, device=self.device)

        ignore_keys = {'mean', 'std', 'epoch', 'val_avg', 'test_avg', 'test_detail'}
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
            if k not in ignore_keys
        }
        model.load_state_dict(clean_state_dict, strict=True)
        model.eval()

        spec = LiTENModelSpec(task_class=task_class, ckpt_path=ckpt_path, is_reg=is_reg, task_names=task_names)
        self._models[task_class] = (model, mean_t, std_t, spec)
        return self._models[task_class]

    @torch.no_grad()
    def predict(self, smiles_list: List[str], task_class: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        model, mean_t, std_t, spec = self._load_task(task_class)

        wanted_cols = columns if columns is not None else spec.task_names
        wanted_cols = [c for c in wanted_cols if c in spec.task_names]

        cached_rows: List[Dict[str, float]] = []
        to_compute: Dict[str, str] = {}
        for smi in smiles_list:
            cached = self._cache_get(task_class, smi)
            # Only use cache if it contains ALL wanted columns
            # Otherwise re-compute the full task set for this SMILES
            if cached is not None and all(c in cached for c in wanted_cols):
                row = {'uid': smi}
                for c in wanted_cols:
                    row[c] = float(cached[c])
                cached_rows.append(row)
            else:
                to_compute[smi] = smi

        computed_df = pd.DataFrame(columns=['uid'] + wanted_cols)
        if to_compute:
            data_list: List[Data] = []
            for uid, smi in to_compute.items():
                mols = _generate_lowest_energy_conformers(smi, num_confs=self.num_confs, top_k=self.top_k_confs)
                for mol in mols:
                    pyg_data = _mol_to_pyg_data(mol, uid=uid)
                    if pyg_data is not None:
                        data_list.append(pyg_data)

            if data_list:
                loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
                all_preds = []
                all_uids: List[str] = []
                for batch in loader:
                    batch = batch.to(self.device)
                    out = model(batch, mean_t, std_t)
                    if not spec.is_reg:
                        out = torch.sigmoid(out)
                    all_preds.append(out.detach().cpu().numpy())
                    for uid in batch.uid:
                        all_uids.append(uid[0] if isinstance(uid, list) else uid)

                preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, len(spec.task_names)))
                df = pd.DataFrame(preds, columns=spec.task_names)
                df['uid'] = all_uids
                df = df[['uid'] + spec.task_names]
                df_agg = df.groupby('uid').mean().reset_index()
                if wanted_cols != spec.task_names:
                    df_agg = df_agg[['uid'] + wanted_cols]
                computed_df = df_agg

            for _, row in computed_df.iterrows():
                uid = str(row['uid'])
                values = {c: float(row[c]) for c in wanted_cols if c in row and pd.notna(row[c])}
                self._cache_set(task_class, uid, values)

        if cached_rows:
            cached_df = pd.DataFrame(cached_rows)
        else:
            cached_df = pd.DataFrame(columns=['uid'] + wanted_cols)

        if computed_df.empty:
            out_df = cached_df
        elif cached_df.empty:
            out_df = computed_df
        else:
            out_df = pd.concat([cached_df, computed_df], ignore_index=True)

        out_df = out_df.drop_duplicates(subset=['uid'], keep='last')
        out_df = out_df.set_index('uid').reindex(smiles_list).reset_index()
        
        # Apply post-processing to convert model outputs to interpretable values
        # This handles properties where the model outputs the "negative" probability
        out_df = self._post_process_predictions(out_df)
        
        return out_df

    def _post_process_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process model predictions to convert to interpretable values.
        
        Some properties (HIA, F20%, F30%, F50%) output the probability of NOT having
        the desired property. We invert these so that higher values mean better property.
        
        The Fu (unbound fraction) is output as log10, we convert back to percentage.
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Probability inversion: model outputs "bad" probability, we want "good"
        invert_cols = ['HIA', 'F20%', 'F30%', 'F50%']
        for col in invert_cols:
            if col in df.columns:
                df[col] = 1.0 - df[col]
        
        # 2. Log还原与百分比转换 for Fu (unbound fraction)
        if 'Fu' in df.columns:
            # Fu is stored as log10, convert back: 10^-log10(Fu) * 100
            df['Fu'] = (10.0 ** -df['Fu']) * 100.0
        
        return df

