from collections import defaultdict
from enum import Enum
from functools import lru_cache
from itertools import chain, count, islice, repeat
import logging
from numbers import Number
from pathlib import Path
import random
import tempfile
import time
from typing import Callable, DefaultDict, Dict, Generator, Hashable, \
    Iterable, List, NewType, Sequence, Set, Tuple, Union
import warnings
from bidict import bidict
import numpy as np
import torch
from torch import Tensor
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from easydict import EasyDict as edict

from ..learner import LearnerCVAEComplex
from ..datasets import ComplexAABatch, ComplexAAItem, ComplexAACollate
from ..utils import StrPath, Vocab, ignore_exception, mol_to_frag, ifn, pretty_kv, download_pbar, complex_to_aa
from ..evaluation import DockingToolBox, QEDSA
from ..public import PUBLIC_DN

DEFAULT_SAVED_DN = PUBLIC_DN / "saved/cvae_complex/20230305_163841_cee4"
DEFAULT_WEIGHT_CHOICE = "latest"
DEFAULT_DEVICE = "cpu"

ChemAssertTypeEnum = Enum('ChemAssertTypeEnum', 'VALIDATE UNIQUE EMBED')
ChemAssertActionEnum = Enum('ChemAssertExpectionEnum', 'WARNING LOGGING RETURN')

@lru_cache(maxsize=1<<16)
def embed_smi_cached(smi: str) -> bool:
    return DockingToolBox.embed_smi(smi)

@lru_cache(maxsize=1<<16)
def __detokenize_cached(seq: Tuple[str, ...]) -> str:
    return mol_to_frag.detokenize(seq)

def detokenize_cached(seq: List[str]) -> str:
    return __detokenize_cached(tuple(seq))

class ChemAssertException(Exception):
    def __init__(self, assert_type: ChemAssertTypeEnum, detail: str=None) -> None:
        if isinstance(assert_type, ChemAssertTypeEnum):
            message = f"Assert {assert_type.name} failed: {detail}"
            self.assert_type = assert_type
        else:
            message = assert_type
        super().__init__(message)

class APICVAEComplex:
    """High level API with chemical inspection
    """

    def __init__(self, 
                 saved_dn: StrPath=DEFAULT_SAVED_DN, weight_choice: str=DEFAULT_WEIGHT_CHOICE,
                 device: str=DEFAULT_DEVICE, 
                 **kwargs) -> None:

        learner_cls = LearnerCVAEComplex
        self.learner = learner_cls.init_inference_from_saved_dn(
            saved_dn, weight_choice, device, **kwargs
        )

    def chem_sample(self, item: ComplexAAItem, batch_size: int=8,
                    assert_types: Set[ChemAssertTypeEnum]=set(ChemAssertTypeEnum),
                    assert_actions: Set[ChemAssertActionEnum]=set(),
                    max_attempts: int=None,
                    max_attempts_exceeded_action: str='raise',
                    desc: str="") -> 'ChemSampler':
        """Target based molecular generation
        """
        sampler = self.learner.sample(item, batch_size)
        x_vocab = self.learner.x_vocab
        return ChemSampler(sampler, x_vocab, 
                           assert_types, assert_actions, max_attempts, 
                           max_attempts_exceeded_action, desc)

    def chem_optimize(self, items: Union[Iterable[ComplexAAItem], ComplexAAItem], 
                      batch_size: int=8,
                      assert_types: Set[ChemAssertTypeEnum]=set(ChemAssertTypeEnum),
                      assert_actions: Set[ChemAssertActionEnum]=set(),
                      max_attempts: int=None,
                      desc: str="<id>") -> 'ChemOptimizer':
        """Target-based molecular optimization

        items: Iterable. If a single item is entered, the optimized molecules 
        will be generated indefinitely

        Note that if ChemAssertTypeEnum.UNIQUE specified, unique is performed 
        for each item.id
        """
        if isinstance(items, ComplexAAItem):
            items = repeat(items)
        collate_fn = self.learner.collate_fn
        optimize_fn = self.learner.optimize
        x_vocab = self.learner.x_vocab
        return ChemOptimizer(items, batch_size, x_vocab, collate_fn, optimize_fn, 
                             assert_types, assert_actions, max_attempts, desc)

    def item_make(self, input_data, input_type='seq') -> 'ItemMaker':
        return ItemMaker(input_data, input_type)

    def mol_evaluate(self, smis: Iterable[str]) -> 'MolEvaluater':
        return MolEvaluater(smis)

class ChemSampler:
    def __init__(self, sampler: Generator[Tuple[List, np.ndarray], None, None], 
                 x_vocab: Vocab, 
                 assert_types: Set[ChemAssertTypeEnum], 
                 assert_actions: Set[ChemAssertActionEnum],
                 max_attempts: int=None,
                 max_attempts_exceeded_action: str='raise',
                 desc: str="") -> None:
        self.sampler = sampler
        self.x_vocab = x_vocab
        self.assert_types = assert_types
        self.assert_actions = assert_actions
        self.max_attempts = max_attempts
        self.max_attempts_exceeded_action = max_attempts_exceeded_action
        self.desc = desc

        assert len(assert_types) == 0 or ChemAssertTypeEnum.VALIDATE in assert_types

        self.smi_history_set = set()
        self.num_attempts = 0
        self.num_success = 0
        self.num_asserts = {e: 0 for e in ChemAssertTypeEnum if e in assert_types}
    
    @property
    def status_dic(self) -> Dict:
        return {
            "num_attempts": self.num_attempts,
            "num_success": self.num_success,
            **{f"not_{e.name.lower()}": v for e, v in self.num_asserts.items()}
        }

    def __iter__(self) -> 'ChemSampler':
        return self
    
    def __next__(self) -> Union[Dict, ChemAssertException]:
        self.num_attempts += 1
        if self.max_attempts is not None and self.num_attempts > self.max_attempts:
            if self.max_attempts_exceeded_action == 'raise':
                raise Exception(f"Exceeded the maximum number of attempts {self.num_attempts}")
            elif self.max_attempts_exceeded_action == 'stop':
                raise StopIteration
        prd_seq, c_attn = next(self.sampler)
        res = edict(seq=None, smi=None, frags=None, attn=c_attn)
        res.seq = prd_seq[:-1]
        try:
            stage = ChemAssertTypeEnum.VALIDATE
            try:
                assert len(res.seq) > 0, "Empty seq"
                res.frags = self.x_vocab.itos(res.seq)
                res.smi = mol_to_frag.detokenize(res.frags)
                assert len(res.smi) > 0, "Empty smi"
            except Exception as err:
                if stage in self.assert_types:
                    raise ChemAssertException(stage, err)

            stage = ChemAssertTypeEnum.UNIQUE
            if stage in self.assert_types:
                if res.smi in self.smi_history_set:
                    raise ChemAssertException(stage, res.smi)
                else:
                    self.smi_history_set.add(res.smi)

            stage = ChemAssertTypeEnum.EMBED
            if stage in self.assert_types:
                if not embed_smi_cached(res.smi):
                    raise ChemAssertException(stage, res.smi)
                
        except ChemAssertException as err:
            self.num_asserts[err.assert_type] += 1
            if ChemAssertActionEnum.WARNING in self.assert_actions:
                warnings.warn(f"{self.desc} -> {repr(err)}")
            if ChemAssertActionEnum.LOGGING in self.assert_actions:
                logging.warning(f"{self.desc} -> {repr(err)}")
            if ChemAssertActionEnum.RETURN in self.assert_actions:
                return err
            else:
                return next(self)
        else:
            self.num_success += 1
            return res
        
class ChemOptimizer:
    def __init__(self, items: Iterable[ComplexAAItem], 
                 batch_size: int, x_vocab: Vocab, 
                 collate_fn: Callable, optimize_fn: Callable, 
                 assert_types: Set[ChemAssertTypeEnum], 
                 assert_actions: Set[ChemAssertActionEnum],
                 max_attempts: int=None,
                 desc: str="<id>") -> None:
        self.items = iter(items)
        self.cid_lst: List[Hashable] = []
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.optimize_fn = optimize_fn
        self.x_vocab = x_vocab
        self.assert_types = assert_types
        self.assert_actions = assert_actions
        self.max_attempts = max_attempts
        self.desc = desc

        assert len(assert_types) == 0 or ChemAssertTypeEnum.VALIDATE in assert_types

        self.smi_history_set_dic: DefaultDict[Hashable, Set] = defaultdict(set)
        self.num_attempts = 0
        self.num_success = 0
        self.num_asserts = {e: 0 for e in ChemAssertTypeEnum if e in assert_types}
    
    @property
    def status_dic(self) -> Dict:
        return {
            "num_attempts": self.num_attempts,
            "num_success": self.num_success,
            **{f"not_{e.name.lower()}": v for e, v in self.num_asserts.items()}
        }

    def ordered_iter(self):
        pend_dic: Dict[Hashable, Union[edict, ChemAssertException]] = dict()
        cursor = 0
        for cid, res in self:
            pend_dic[cid] = res
            while cursor < len(self.cid_lst) and self.cid_lst[cursor] in pend_dic:
                yield self.cid_lst[cursor], pend_dic.pop(self.cid_lst[cursor])
                cursor += 1

    def __iter__(self):
        cx_item_lst: List[ComplexAAItem] = []
        while True:
            new_item_lst = list(islice(self.items, self.batch_size - len(cx_item_lst)))
            self.cid_lst += [x.id for x in new_item_lst]
            cx_item_lst = [*cx_item_lst, *new_item_lst]
            if len(cx_item_lst) == 0:
                break
            batch = self.collate_fn(cx_item_lst)
            batch_result_od = self.optimize_fn(batch, variational=True)
            failed_cx_item_lst = []
            for cx_item in cx_item_lst:
                result = batch_result_od[cx_item.id]
                self.num_attempts += 1
                if self.max_attempts is not None and self.num_attempts > self.max_attempts:
                    raise Exception(f"Exceeded the maximum number of attempts {self.num_attempts}")
                
                res = edict(seq=None, smi=None, frags=None, attn=result.c_attn)
                res.seq = result.prd_seq[:-1]
                try:
                    stage = ChemAssertTypeEnum.VALIDATE
                    try:
                        assert len(res.seq) > 0, "Empty seq"
                        res.frags = self.x_vocab.itos(res.seq)
                        res.smi = mol_to_frag.detokenize(res.frags)
                        assert len(res.smi) > 0, "Empty smi"
                    except Exception as err:
                        if stage in self.assert_types:
                            raise ChemAssertException(stage, err)

                    stage = ChemAssertTypeEnum.UNIQUE
                    if stage in self.assert_types:
                        if res.smi in self.smi_history_set_dic[cx_item.id]:
                            raise ChemAssertException(stage, res.smi)
                        else:
                            self.smi_history_set_dic[cx_item.id].add(res.smi)

                    stage = ChemAssertTypeEnum.EMBED
                    if stage in self.assert_types:
                        if not embed_smi_cached(res.smi):
                            raise ChemAssertException(stage, res.smi)
                        
                except ChemAssertException as err:
                    self.num_asserts[err.assert_type] += 1
                    failed_cx_item_lst.append(cx_item)

                    if ChemAssertActionEnum.WARNING in self.assert_actions:
                        warnings.warn(f"{self.desc.replace('<id>', cx_item.id)} -> {repr(err)}")
                    if ChemAssertActionEnum.LOGGING in self.assert_actions:
                        logging.warning(f"{self.desc.replace('<id>', cx_item.id)} -> {repr(err)}")
                    if ChemAssertActionEnum.RETURN in self.assert_actions:
                        yield cx_item.id, err
                else:
                    self.num_success += 1
                    yield cx_item.id, res

            cx_item_lst = failed_cx_item_lst

_TSMILES = NewType('_TSMILES', str)
_TFitness = Dict[_TSMILES, Dict[str, Number]]
_TPopuSA = List[_TSMILES]

class APICVAEComplex4SA(APICVAEComplex):
    """High level API for SA

    - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html
    - https://en.wikipedia.org/wiki/Simulated_annealing
    - https://oi-wiki.org/misc/simulated-annealing/
    """

    TSMILES = _TSMILES
    TFitness= _TFitness
    TPopuSA = _TPopuSA

    def init_popu(self,
                  c_item: ComplexAAItem, fitness_fn: Callable[[_TPopuSA], _TFitness],
                  popu_size: int=200, T0: float=100, D: float=0.8, 
                  batch_size: int=8, random_seed: int=20230403) -> None:
        """Initialize Population
        """

        self.c_item = c_item
        self.fitness_fn = fitness_fn
        # Return dict: {smi: {tox: 12, qed:1, scaled: 0.7}}
        # For scaled: The larger, the better 
        self.popu_size = popu_size
        self.batch_size = batch_size

        self.T0 = T0
        self.D = D
        self.Tf = None

        self.r = np.random.default_rng(random_seed)

        self.popu_lst: _TPopuSA = list()
        self.seq_dic: Dict[_TSMILES, List[int]] = dict()
        self.popu_lst_history: Dict[str, _TPopuSA] = dict()

    def generate_popu(self, desc="Initialize Population") -> int:
        sampler = self.chem_sample(self.c_item, self.batch_size)
        sample_res_gen = islice(sampler, self.popu_size)
        pbar = tqdm(sample_res_gen, total=self.popu_size, desc=desc)
        for res in pbar:
            res = edict(res)
            self.popu_lst.append(res.smi)
            self.seq_dic[res.smi] = res.seq
        assert len(self.popu_lst) == self.popu_size
        return sampler.num_attempts

    def evaluate_popu(self, popu_lst: _TPopuSA=None) -> Tuple[_TSMILES, _TFitness]:
        popu_lst = ifn(popu_lst, self.popu_lst)
        fitness_dic = self.fitness_fn(popu_lst)
        sorted_lst = sorted(fitness_dic, key=lambda k: fitness_dic[k]["scaled"])
        return sorted_lst, fitness_dic

    def update_popu(self, T: float, desc="Update Population") -> Tuple[int, int]:
        fitness_dic = self.fitness_fn(self.popu_lst)
        E_popu = [fitness_dic[smi]["scaled"] for smi in self.popu_lst]

        items = self.get_popu_items()
        optimizer = self.chem_optimize(items, self.batch_size)
        pbar = tqdm(optimizer, total=self.popu_size, desc=desc)
        neig_lst = [None] * self.popu_size
        for cid, res in pbar:
            res = edict(res)
            neig_lst[cid] = res.smi
            self.seq_dic[res.smi] = res.seq
        assert None not in neig_lst

        fitness_dic = self.fitness_fn(neig_lst)
        E_neig = [fitness_dic[smi]["scaled"] for smi in neig_lst]

        # Minimize
        E_delta = np.array(E_neig) - np.array(E_popu)
        with np.errstate(over='ignore'):
            trans_mask = (E_delta < 0) | (1 / (1 + np.exp(E_delta / T)) > self.r.random(self.popu_size))
        self.popu_lst = [new if trans else old for old, new, trans in zip(self.popu_lst, neig_lst, trans_mask)]
        return optimizer.num_attempts, trans_mask.sum()

    def log_popu(self, epoch: int) -> str:
        popu_lst = self.popu_lst.copy()
        self.popu_lst_history[epoch] = popu_lst
        sorted_lst, fitness_dic = self.evaluate_popu(popu_lst)
        score_lst_dic = defaultdict(list)
        for score_dic in fitness_dic.values():
            for k, v in score_dic.items():
                score_lst_dic[k].append(v)
        res_dic = {k: np.mean(v) for k, v in score_lst_dic.items()}
        res_dic["uniqueness"] = len(set(popu_lst)) / len(popu_lst)
        res_str = f"Epoch.{epoch:<3} -> {pretty_kv(res_dic, ndigits=5)}"
        return res_str

    def top_popu(self, top_k: int=-1) -> _TFitness:
        popu_lst = list(set(chain.from_iterable(self.popu_lst_history.values())))
        sorted_lst, fitness_dic = self.evaluate_popu(popu_lst)
        top_lst = sorted_lst[:top_k]
        res_lst = [{"smi": smi, **fitness_dic[smi]} for smi in top_lst]
        return res_lst

    def run(self, epochs: int, top_k: int=-1) -> List[_TFitness]:
        # T0*(D^k) = Tf
        self.Tf = self.T0 * (self.D ** (epochs - 1))
        num_attempts = self.generate_popu()
        log_res = self.log_popu(0)
        logging.info(f"{log_res}, succ: {self.popu_size/num_attempts*100:.3f}%, "
                     f"T0: {self.T0:.5f}, Tf: {self.Tf:.5f}, D: {self.D:.5f}")

        T = self.T0
        for epoch in range(epochs):
            num_attempts, num_trans = self.update_popu(T)
            log_res = self.log_popu(epoch+1)
            logging.info(f"{log_res}, succ: {self.popu_size/num_attempts*100:.3f}%, trans: {num_trans}, T: {T:.5f}")
            T *= self.D
        res_lst = self.top_popu(top_k)
        return res_lst

    def get_popu_items(self):
        for idx, smi in enumerate(self.popu_lst):
            item = ComplexAAItem(
                id=idx,
                x=[self.learner.x_vocab.special_idx.sos, 
                *self.seq_dic[smi][:self.learner.x_max_len - 2], 
                self.learner.x_vocab.special_idx.eos],
                c=self.c_item.c)
            yield item

    def info(self) -> Dict:
        dic = dict()
        dic['popu'] = [self.popu_lst_history[i] for i in range(len(self.popu_lst_history))]
        _, dic['fitness'] = self.evaluate_popu(list(set(chain.from_iterable(dic['popu']))))
        dic['para'] = dict(T0=self.T0, D=self.D, Tf=self.Tf)
        return dic
    
ITEM_MAKER_INPUT_TYPES = {'seq', 'pdb', 'url', 'pdb_fn', 'pdb_id'}
class ItemMaker:
    def __init__(self, input_data, input_type) -> None:
        self.d = defaultdict(lambda: None)
        if input_type == 'seq':
            self.item = self._from_seq(input_data)
        elif input_type == 'pdb':
            self.item = self._from_pdb(input_data)
        elif input_type == 'url':
            self.item = self._from_url(input_data)
        elif input_type == 'pdb_fn':
            self.item = self._from_pdb_fn(input_data)
        elif input_type == 'pdb_id':
            self.item = self._from_pdb_id(input_data)
        else:
            raise Exception(f'Unknow input type: {input_type}')
        
    def _from_seq(self, seq: str):
        assert isinstance(seq, str)
        item_id = str(int(time.time()*1000))
        if self.d['aa'] is not None: item_id += self.d['aa'].id
        return ComplexAAItem(id=item_id, x=[], c=seq)

    def _from_aa(self, aa: complex_to_aa.ComplexAAExtract):
        assert isinstance(aa, complex_to_aa.ComplexAAExtract)
        self.d['aa'] = aa
        return self._from_seq(aa.seq)

    def _from_pdb(self, pdb: bytes):
        assert isinstance(pdb, bytes)
        self.d['pdb'] = pdb
        with tempfile.TemporaryDirectory() as tmpdir:
            fn = Path(tmpdir) / "rec.pdb"
            with open(fn, 'wb') as f:
                f.write(pdb)
            aa = complex_to_aa.extract(fn)
            return self._from_aa(aa)

    def _from_pdb_fn(self, pdb_fn: StrPath):
        pdb_fn = Path(pdb_fn)
        assert pdb_fn.exists()
        self.d['pdb_fn'] = pdb_fn
        with open(pdb_fn, 'rb') as f:
            pdb = f.read()
        return self._from_pdb(pdb)

    def _from_url(self, url: str):
        assert isinstance(url, str)
        self.d['url'] = url
        pdb = bytes(download_pbar(url))
        return self._from_pdb(pdb)

    def _from_pdb_id(self, pdb_id: str):
        assert isinstance(pdb_id, str)
        self.d['pdb_id'] = pdb_id
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        return self._from_url(url)
    
class MolEvaluater:
    def __init__(self, smis: Iterable[str]) -> None:
        self.smis = smis
    def __iter__(self):
        for smi in self.smis:
            qedsa = QEDSA(smi)
            result = {
                "qed": qedsa.qed(),
                "sa": qedsa.sa(),
                "logp": qedsa.logp()
            }
            lipinski, rules = qedsa.lipinski()
            result["lipinski"] = lipinski
            for i in range(5):
                result[f"lipinski_{i+1}"] = int(rules[i])
            yield result
