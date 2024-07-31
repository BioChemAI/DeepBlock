import sys
from pathlib import Path
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from ..utils import ignore_exception, rdkit_log_handle, vin
from ..exceptions import RDKitException

try:
    sys.path.append(str(Path(Chem.RDConfig.RDContribDir)/ 'SA_Score'))
    from sascorer import calculateScore  # pyright: ignore[reportMissingImports]
except ImportError as err:
    warnings.warn(f"Unable to calculate SA Score. "
                  f"Because your RDKit Contrib seems incomplete. Please refer to "
                  f"https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score\n"
                  f"{repr(err)}")

class QEDSA():
    def __init__(self, smiles) -> None:
        with rdkit_log_handle() as rdLog:
            self.rdmol = Chem.MolFromSmiles(smiles)
            if self.rdmol is None:
                raise RDKitException(rdLog())

    @ignore_exception
    @rdkit_log_handle()
    def qed(self) -> float:
        _qed = Descriptors.qed(self.rdmol)
        return _qed
    
    @ignore_exception
    @rdkit_log_handle()
    def sa(self, is_norm=False) -> float:
        _sa = calculateScore(self.rdmol)
        if is_norm:
            _sa = round((10 - _sa) / 9, 2)
        return _sa

    @ignore_exception
    @rdkit_log_handle()
    def lipinski(self, n_rules=5):
        rule_to_idx = {5: range(5), 4: range(4), 3: (0,1,3)}
        vin(n_rules, rule_to_idx.keys())
        rules = (
            Descriptors.ExactMolWt(self.rdmol) < 500,
            Lipinski.NumHDonors(self.rdmol) <= 5,
            Lipinski.NumHAcceptors(self.rdmol) <= 10,
            -2 <= Crippen.MolLogP(self.rdmol) <= 5,
            Chem.rdMolDescriptors.CalcNumRotatableBonds(self.rdmol) <= 10
        )
        _lipinski = sum(rules[i] for i in rule_to_idx[n_rules])
        return _lipinski, rules

    @ignore_exception
    @rdkit_log_handle()
    def logp(self) -> float:
        _logp = Crippen.MolLogP(self.rdmol)
        return _logp
        