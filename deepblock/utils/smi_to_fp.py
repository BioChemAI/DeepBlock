from typing import Callable, Tuple
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from . import Mol, rdkit_mol_decorator

_TMolFpFunc = Callable[[Mol], Tuple[int]]
_TSmiFpFunc = Callable[[str], Tuple[int]]
_TMol2FpDec = Callable[[_TMolFpFunc], _TSmiFpFunc]
fp_decorator: _TMol2FpDec = rdkit_mol_decorator

@fp_decorator
def maccs(mol: Mol) -> Tuple[int]:
    fp = MACCSkeys.GenMACCSKeys(mol)
    return tuple(fp.ToList())
setattr(maccs, 'fp_length', 127)
