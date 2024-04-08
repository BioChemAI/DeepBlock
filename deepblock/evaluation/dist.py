from itertools import combinations
from typing import List
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import Mol
import numpy as np

TFingerprint = DataStructs.cDataStructs.ExplicitBitVect

class Dist:
    @classmethod
    def fp(cls, mol: Mol) -> TFingerprint:
        return Chem.RDKFingerprint(mol)

    @classmethod
    def scaffold_fp(cls, mol: Mol) -> TFingerprint:
        mol = MurckoScaffold.GetScaffoldForMol(mol)
        return cls.fp(mol)

    @classmethod
    def _similarity(cls, fp1: TFingerprint, fp2: TFingerprint):
        return DataStructs.FingerprintSimilarity(fp1, fp2)

    @classmethod
    def similarity(cls, fps: List[TFingerprint]):
        similarity_lst = [
            cls._similarity(fp_x, fp_y) for fp_x, fp_y in combinations(fps, 2)
        ]
        return float(np.mean(similarity_lst))

    @classmethod
    def ref_similarity(cls, ref_fp, prd_fps: List[TFingerprint]):
        similarity_lst = [
            cls._similarity(ref_fp, prd_fp) for prd_fp in prd_fps
        ]
        return float(np.mean(similarity_lst))
    
    @classmethod
    def novelty_fp(cls, 
                   train_fps: List[TFingerprint], 
                   prd_fps: List[TFingerprint], threshold=0.4):
        count = 0
        for fp in prd_fps:
            for train_fp in train_fps:
                similarity = cls._similarity(train_fp, fp)
                if similarity > threshold:
                    count += 1
                    break
        return 1 - count / len(prd_fps)

    @classmethod
    def novelty_smi(cls, 
                    train_smi_lst: List[str], 
                    prd_smi_lst: List[str]):
        prd_smi_lst = [x for x in prd_smi_lst if x]
        novel_smi_set = set(prd_smi_lst) - set(train_smi_lst)
        count = sum((x in novel_smi_set) for x in prd_smi_lst)
        return count / len(prd_smi_lst)
