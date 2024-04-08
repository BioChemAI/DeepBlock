"""
A simple vocab class:

```
vocab = Vocab.from_seq(['a', 'b', 'c', 'd'])
vocab.itos_lst = ['<sos>', '<eos>', '<unk>', '<pad>', 'a', 'b', 'c', 'd']
vocab.stoi('a') # 4
vocab.itos(6) # 'c'
vocab.itos([7, 5]) # ['d', 'b']
vocab.stoi(['c', ['a', 'd']]) # [6, [4, 7]]
vocab.itos(vocab.stoi('e')) # <unk>
```
"""

from collections import Counter
from dataclasses import asdict, astuple, dataclass
from typing import Dict, List, Sequence, Iterable

@dataclass
class VocabSpecialSymbol():
    sos: str = "<sos>"
    eos: str = "<eos>"
    unk: str = "<unk>"
    pad: str = "<pad>"

@dataclass
class VocabSpecialIndex():
    sos: int
    eos: int
    unk: int
    pad: int

class Vocab():
    def __init__(self, itos_lst: List[str], specials: Dict={}):
        self.itos_lst = itos_lst
        self.itos_set = set(self.itos_lst)
        assert len(self.itos_lst) == len(self.itos_set), "Duplicate in dictionary!"
        self.stoi_dic = {s: i for i, s in enumerate(self.itos_lst)}
        self.special_sym = VocabSpecialSymbol(**specials)
        self.special_idx = self.special_stoi(self.special_sym)

    @classmethod
    def from_seq(cls, seq: Sequence[str], min_freq=1, specials={}):
        c = Counter(seq)
        special_sym = VocabSpecialSymbol(**specials)
        specials = asdict(special_sym)
        itos_lst = list(specials.values()) + \
            sorted([elem for elem, cnt in c.items() if cnt >= min_freq])
        return cls(itos_lst, specials)

    def to_dict(self):
        return dict(itos_lst=self.itos_lst, specials=asdict(self.special_sym))

    def special_stoi(self, special_sym: VocabSpecialSymbol) -> VocabSpecialIndex:
        return VocabSpecialIndex(**{k: self.stoi_dic[v] for k, v in asdict(special_sym).items()})

    def coverage(self, seqs: Iterable[Sequence[str]]):
        return sum(self.is_coverage(seq) for seq in seqs)

    def is_coverage(self, seq: Sequence[str]):
        return len(set(seq) - self.itos_set) == 0

    def itos(self, i: int):
        if isinstance(i, (list, tuple)):
            return list(map(self.itos, i))
        else:
            return self.itos_lst[i]

    def stoi(self, s: str):
        if isinstance(s, (list, tuple)):
            return list(map(self.stoi, s))
        else:
            return self.stoi_dic.get(s, self.special_idx.unk)

    def __len__(self) -> int:
        return len(self.itos_lst)

    def __getitem__(self, idx) -> str:
        return self.itos_lst[idx]

    def __add__(self, other):
        assert isinstance(other, Vocab)
        avoid_set = set(self.itos_lst + list(astuple(other.special_sym)))
        new_lst = [s for s in other.itos_lst if s not in avoid_set]
        new_vocab = self.__class__(
            itos_lst=self.itos_lst + new_lst, 
            specials=asdict(self.special_sym))
        return new_vocab
