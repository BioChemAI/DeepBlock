from typing import Tuple
from rdkit import Chem

from ..exceptions import IllegalSMILESException, ReconstructException
from . import rdkit_log_handle
from .brics_then_ordered_dfs import SeqBRICS


def tokenize(smi: str, return_act_smi: bool=False, abbr=True) -> Tuple[str]:
    """Cutting a molecule into fragment sequences.

    Args:
        smi (str): SMILES of a molecule.
        return_act_smi (bool, optional): Whether to return the molecules that passed 
        the comparison after removing some chiral markers. Defaults to False.

    Raises:
        IllegalSMILESException: Illegal SMILES of molecule.
        ReconstructException: Reconstruction comparison failed.

    Returns:
        Tuple[str]: SMILES sequence of fragments.
    """
    with rdkit_log_handle() as rdLog:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            raise IllegalSMILESException(rdLog())

        mol = SeqBRICS.mol_to_mol_by_smi(mol)

        smi1 = Chem.MolToSmiles(mol)
        seq = tuple(SeqBRICS.fragment(mol, abbr=abbr))
        smi2 = detokenize(seq, abbr)

        clean_smi = lambda smi: SeqBRICS.smi_to_smi_by_mol(smi.replace("/", "").replace("\\", ""))

        if smi1 == smi2:
            act_smi = smi1
        elif clean_smi(smi1) == clean_smi(smi2):
            act_smi = smi2
        else:
            raise ReconstructException(f"Unable to reconstruct correctly: {smi1} => {seq} => {smi2}")

        if return_act_smi:
            return seq, act_smi
        else:
            return seq

def detokenize(seq: Tuple[str, ...], abbr=True) -> str:
    """Reconstruct fragment sequences into one molecule.

    Args:
        seq (Tuple[str, ...]): SMILES sequence of fragments.

    Returns:
        str: SMILES of a molecule.
    """
    with rdkit_log_handle() as rdLog:
        mol = SeqBRICS.reconstruct(list(seq), abbr)
        smi = Chem.MolToSmiles(mol)
        return smi
