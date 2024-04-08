import re
from typing import List, Tuple

from ..exceptions import ReconstructException
from . import rdkit_log_handle


def tokenize(smi: str) -> Tuple[str]:
    seq = tuple(re.findall("%\d\d|\[.+?\]|Cl|Br|.", smi))
    smi1 = detokenize(seq)
    if smi == smi1:
        return seq
    else:
        raise ReconstructException(f"Unable to reconstruct correctly: {smi} => {seq} => {smi1}")


def detokenize(seq: Tuple[str, ...]) -> str:
    smi = ''.join(seq)
    return smi
