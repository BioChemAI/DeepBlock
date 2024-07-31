from itertools import groupby
import hashlib
import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Callable, Hashable, List, Dict, Sequence, Union
from dataclasses import dataclass, field
from halo import Halo

from esm.data import Alphabet

from . import CrossDockedDataset, ChEMBLDataset, PDBbindDataset
from ..utils import Vocab, VocabSpecialIndex, gc_disabled, ifn, norm_rel_fn, \
    pick_by_idx, pretty_dataclass_repr, sorted_seqs, split_pro_to_idx, auto_load
from ..utils.complex_to_aa import ComplexAAExtract

# 3 possibilities
# ComplexAAItem(id, x)              -> ComplexAABatch(total, id, x, x_len)
# ComplexAAItem(id, x, c)           -> ComplexAABatch(total, id, x, x_len, c, c_len)
# ComplexAAItem(id, x, c, c_rel)    -> ComplexAABatch(total, id, x, x_len, c, c_len, c_rel)

@dataclass
class ComplexAAItem:
    id: Hashable                   # Hashable ID
    x: Sequence[int]               # Ligand idx sequence -> [0, 2, 3, 1] with sos
    c: str = None                  # Protein amino acid sequence -> 'MKTVRQERLKSIVR' without sos
    c_rel: Sequence[float] = None  # Relevance between pocket and residue (sum 1)-> [0.1, 0.2, 0.1] without sos
    c_hash: Hashable = field(init=False)

    def __post_init__(self):
        if self.c_rel is not None:
            if not len(self.c_rel) == len(self.c):
                raise ValueError("The length of the relevance needs to be equal to the sequence")
            if not np.isclose(np.sum(self.c_rel), 1):
                raise ValueError("The sum of relevance needs to be 1")
        if self.c is not None:
            self.c_hash = hashlib.sha256(self.c.encode()).digest()

    def __repr__(self) -> str:
        return pretty_dataclass_repr(self)

@dataclass
class ComplexAABatch:
    total: int
    id: List[Hashable]
    x: Tensor               # with sos
    x_len: Tensor
    c: Tensor = None        # with sos
    c_len: Tensor = None
    c_rel: Tensor = None    # with sos
    c_hash: List[Hashable] = None
    
    def __len__(self):
        return self.total

    def __repr__(self) -> str:
        return pretty_dataclass_repr(self)

class ComplexAACollate:
    def __init__(self, x_special_idx: VocabSpecialIndex, esm_alphabet: Alphabet, 
                 batch_first=True, sort=False):
        self.x_special_idx = x_special_idx
        self.esm_alphabet = esm_alphabet
        self.sort = sort
        self.batch_first = batch_first

        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

    def __call__(self, batch: List[ComplexAAItem]) -> ComplexAABatch:
        if self.sort:
            sorted_idx_lst = sorted_seqs(
                (t.x for t in batch), (t.c for t in batch), return_idx=True)
            batch = pick_by_idx(batch, idx_lst=sorted_idx_lst)

        x_seqs = [torch.tensor(t.x) for t in batch]
        _batch = ComplexAABatch(
            id=[t.id for t in batch],
            x=pad_sequence(x_seqs, padding_value=self.x_special_idx.pad, batch_first=self.batch_first),
            x_len=torch.tensor([len(t.x) for t in batch]),
            total=len(batch)
        )

        _count = sum(t.c is not None for t in batch)
        if _count == len(batch):
            c_data = [(t.id, t.c) for t in batch]
            _, _, _batch.c = self.esm_batch_converter(c_data)
            _batch.c_len = (_batch.c != self.esm_alphabet.padding_idx).sum(1)
            _batch.c_hash = [t.c_hash for t in batch]
        elif _count != 0:
            raise Exception("There are exotic flowers mixed in the batch [t.c]")

        _count = sum(t.c_rel is not None for t in batch)
        if _count == len(batch):
            # torch.from_numpy is very fast
            c_rel_seqs = [torch.from_numpy(t.c_rel)
                if isinstance(t.c_rel, np.ndarray)
                else torch.tensor(t.c_rel) for t in batch]
            _batch.c_rel = torch.cat((
                torch.zeros(len(batch), 1) if self.batch_first else torch.zeros(1, len(batch)),
                pad_sequence(c_rel_seqs, padding_value=0, batch_first=self.batch_first),
                torch.zeros(len(batch), 1) if self.batch_first else torch.zeros(1, len(batch)),
                ), dim=1 if self.batch_first else 0)
            assert _batch.c_rel.shape == _batch.c.shape
        elif _count != 0:
            raise Exception("There are exotic flowers mixed in the batch [t.c_rel]")

        return _batch


class ComplexAADataset(Dataset):
    def __init__(self, 
                d: Union[ChEMBLDataset, CrossDockedDataset, PDBbindDataset],
                rel_fn: Callable[[np.ndarray], np.ndarray]=norm_rel_fn,
                x_vocab: Vocab = None,
                unique_xc_seq: bool=True, assert_x_in_vocab: bool=True, only_known: bool=True,
                x_max_len: int=None, c_max_len: int=None,
                split_idx_dic: Dict[str, List[int]]=None, split_pro_dic: Dict[str, float]=None, split_key: str=None,
                is_dev=False, sort=True, transform=None):

        desc = (f'Initialize {self.__class__.__name__} '
                f'from {d.__class__.__name__}'
                f'@{ifn(split_key, "all")}')
        spinner = Halo(text=desc)
        spinner.start()

        self.rel_fn = rel_fn

        self.cleaning_status = {}
        self.raw_lst = []

        spinner.text = desc + ' (1/2) -> source'
        if isinstance(d, CrossDockedDataset):
            meta_lst: List[Dict] = d.source("meta")
            with gc_disabled(mem_threshold=32):
                complex_to_aa_dic: Dict[str, ComplexAAExtract] = d.source("complex_to_aa")
                mol_to_frag_dic: Dict[str, Sequence[str]] = d.source("mol_to_frag")
                self.x_vocab: Vocab = ifn(x_vocab, d.source("frag_vocab"))

            spinner.text = desc + ' (2/2) -> process'

            self.cleaning_status["original_meta"] = len(meta_lst)
            if only_known:
                meta_lst = list(filter(lambda meta: meta.split in {"train", "test"}, meta_lst))
                self.cleaning_status["only_known"] = len(meta_lst)
    
            for meta in meta_lst:
                _cond = (
                    meta.id in complex_to_aa_dic,
                    meta.smi in mol_to_frag_dic,
                )
                if not all(_cond): continue

                x_seq = mol_to_frag_dic[meta.smi]['frag_seq']
                _cond = (
                    not assert_x_in_vocab,
                    self.x_vocab.is_coverage(x_seq),
                )
                if not any(_cond): continue

                aa = complex_to_aa_dic[meta.id]
                if len(aa.seq) == 0: continue

                self.raw_lst.append(dict(
                    id=meta.id, 
                    smi=meta.smi, 
                    x_seq=x_seq, c_seq=aa.seq,
                    c_dist=aa.pocket_dist)
                )
            self.cleaning_status["legaliy"] = len(self.raw_lst)

            if unique_xc_seq:
                # self.raw_lst = unique_by_key(self.raw_lst, 'x_seq', 'c_seq')
                # Unique with minimum RMSD value
                _rmsd_dic = {meta.id: meta.rmsd for meta in meta_lst}
                _f = lambda x: (tuple(x['x_seq']), tuple(x['c_seq']))
                _lst = sorted(self.raw_lst, key=_f)
                _lst = groupby(_lst, _f)
                _lst = map(lambda g: min(g[1], key=lambda x: _rmsd_dic[x['id']]), _lst)
                self.raw_lst = list(_lst)
            self.cleaning_status["unique_xc_seq"] = len(self.raw_lst)

            if split_key:
                if split_key == "test":
                    _id_set = set(meta.id for meta in meta_lst if meta.split == "test")
                    self.raw_lst = list(filter(lambda x: x['id'] in _id_set, self.raw_lst))
                else:
                    _id_set = set(meta.id for meta in meta_lst if meta.split == "train")
                    self.raw_lst = list(filter(lambda x: x['id'] in _id_set, self.raw_lst))

                    if not split_idx_dic:
                        split_idx_dic = split_pro_to_idx(split_pro_dic, total=len(self.raw_lst), seed=20230226)
                    pick_by_idx(
                        self.raw_lst,
                        idx_lst=split_idx_dic[split_key], inplace=True)
                self.cleaning_status[f"split_{split_key}"] = len(self.raw_lst)

        elif isinstance(d, ChEMBLDataset):
            meta_lst: List[Dict] = d.source("meta")
            mol_to_frag_dic: Dict[str, Sequence[str]] = d.source("mol_to_frag")
            self.x_vocab: Vocab = ifn(x_vocab, d.source("frag_vocab"))

            spinner.text = desc + ' (2/2) -> process'

            self.cleaning_status["original_meta"] = len(meta_lst)
    
            for meta in meta_lst:
                _cond = (
                    meta.smi in mol_to_frag_dic,
                )
                if not all(_cond): continue

                x_seq = mol_to_frag_dic[meta.smi]['frag_seq']
                _cond = (
                    not assert_x_in_vocab,
                    self.x_vocab.is_coverage(x_seq)
                )
                if not any(_cond): continue

                self.raw_lst.append(dict(
                    id=meta.id, smi=meta.smi, 
                    x_seq=x_seq)
                )
            self.cleaning_status["legaliy"] = len(self.raw_lst)

            if split_key:
                if not split_idx_dic:
                    split_idx_dic = split_pro_to_idx(split_pro_dic, total=len(self.raw_lst), seed=20230226)
                pick_by_idx(
                    self.raw_lst,
                    idx_lst=split_idx_dic[split_key], inplace=True)
                self.cleaning_status[f"split_{split_key}"] = len(self.raw_lst)

        elif isinstance(d, PDBbindDataset):
            meta_lst: List[Dict] = d.source("meta")
            with gc_disabled(mem_threshold=32):
                complex_to_aa_dic: Dict[str, ComplexAAExtract] = d.source("complex_to_aa")
                mol_to_frag_dic: Dict[str, Sequence[str]] = d.source("mol_to_frag")
                self.x_vocab: Vocab = ifn(x_vocab, d.source("frag_vocab"))

            spinner.text = desc + ' (2/2) -> process'
            self.cleaning_status["original_meta"] = len(meta_lst)
    
            for meta in meta_lst:
                _cond = (
                    meta.id in complex_to_aa_dic,
                    meta.smi in mol_to_frag_dic,
                )
                if not all(_cond): continue

                x_seq = mol_to_frag_dic[meta.smi]['frag_seq']
                _cond = (
                    not assert_x_in_vocab,
                    self.x_vocab.is_coverage(x_seq),
                )
                if not any(_cond): continue

                aa = complex_to_aa_dic[meta.id]
                if len(aa.seq) == 0: continue

                self.raw_lst.append(dict(
                    id=meta.id, 
                    smi=meta.smi, 
                    x_seq=x_seq, c_seq=aa.seq,
                    c_dist=aa.pocket_dist)
                )
            self.cleaning_status["legaliy"] = len(self.raw_lst)

            if unique_xc_seq:
                # self.raw_lst = unique_by_key(self.raw_lst, 'x_seq', 'c_seq')
                # Unique with refined -> high resolution -> latest release year
                _comp_dic = {meta.id: (
                    0 if meta.refined else 1, meta.resolution, -meta.release_year) for meta in meta_lst}
                _f = lambda x: (tuple(x['x_seq']), tuple(x['c_seq']))
                _lst = sorted(self.raw_lst, key=_f)
                _lst = groupby(_lst, _f)
                _lst = map(lambda g: min(g[1], key=lambda x: _comp_dic[x['id']]), _lst)
                self.raw_lst = list(_lst)
            self.cleaning_status["unique_xc_seq"] = len(self.raw_lst)

            if split_key:
                assert "test" not in split_pro_dic, "test should not be in split_pro_dic"
                pick_set_dic = d.source("pick_set")

                if split_key == "test":
                    _id_set = set(pick_set_dic["test"])
                    self.raw_lst = list(filter(lambda x: x['id'] in _id_set, self.raw_lst))
                else:
                    _id_set = set(pick_set_dic["train"])
                    self.raw_lst = list(filter(lambda x: x['id'] in _id_set, self.raw_lst))

                    if not split_idx_dic:
                        split_idx_dic = split_pro_to_idx(split_pro_dic, total=len(self.raw_lst), seed=20240723)
                    pick_by_idx(
                        self.raw_lst,
                        idx_lst=split_idx_dic[split_key], inplace=True)
                self.cleaning_status[f"split_{split_key}"] = len(self.raw_lst)

        else:
            raise Exception(f"Unkown dataset: {d.__class__.__name__}")

        # Abobe: self.raw_lst, self.x_vocab

        if is_dev:
            pick_by_idx(
                self.raw_lst,
                idx_lst=range(min(8<<10, len(self.raw_lst))), inplace=True)
            self.cleaning_status["dev"] = len(self.raw_lst)

        if sort:
            sorted_idx_lst = sorted_seqs(
                [x['x_seq'] for x in self.raw_lst], 
                return_idx=True)
            pick_by_idx(
                self.raw_lst,
                idx_lst=sorted_idx_lst, inplace=True)

        # x_max_len also acts on '<eos>' and '<sos>'
        self.x_max_len = x_max_len
        self.c_max_len = c_max_len

        if self.x_max_len is not None:
            assert self.x_max_len > 2, "x_max_len should > 2"
        if self.c_max_len is not None:
            assert self.c_max_len > 2, "c_max_len should > 2"

        self.transform = transform

        self.cleaning_status["finally"] = len(self.raw_lst)

        spinner.succeed(desc + f" -> Done!")

    @property
    def id_smi_dic(self):
        dic = {x['id']: x['smi'] for x in self.raw_lst}
        return dic

    def id_to_idx(self, cid: Hashable):
        idx = next(i for i, x in enumerate(self.raw_lst) if x['id'] == cid)
        return idx

    def __len__(self):
        return len(self.raw_lst)

    def __getitem__(self, idx: int) -> ComplexAAItem:
        _dic = self.raw_lst[idx]

        x_token = ['<sos>', *_dic['x_seq'][:self.x_max_len - 2], '<eos>']
        c_dist = _dic.get('c_dist', None)
        if c_dist is None:
            c_rel = None
        else:
            c_rel = self.rel_fn(c_dist)

        item = ComplexAAItem(
            id=_dic['id'],
            x=self.x_vocab.stoi(x_token),
            c=_dic.get('c_seq', None),
            c_rel=c_rel
        )
        if self.transform:
            item = self.transform(item)
        return item
    