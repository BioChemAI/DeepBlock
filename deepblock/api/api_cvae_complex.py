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
from ..utils import StrPath, Vocab, ignore_exception, mol_to_frag, ifn, pretty_kv, download_pbar, \
    complex_to_aa, use_path, norm_rel_fn, chunked
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
                    desc: str="",
                    use_groundtruth_rel: bool=False,
                    use_force_mean: bool=False) -> 'ChemSampler':
        """Target based molecular generation
        """
        sampler = self.learner.sample(item, batch_size, 
                                      use_groundtruth_rel=use_groundtruth_rel, 
                                      use_force_mean=use_force_mean)
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

    def item_make(self, input_data, input_type='seq',
                  input_chain_id: str=None,
                  groundtruth_ligand_data: str=None,
                  groundtruth_ligand_type: str=None) -> 'ItemMaker':
        return ItemMaker(input_data, input_type, input_chain_id,
                         groundtruth_ligand_data, groundtruth_ligand_type)

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
        
class APICVAEComplex4MSO:
    """High level API for MSO
    """

    def __init__(self, 
                 saved_dn: StrPath=DEFAULT_SAVED_DN, weight_choice: str=DEFAULT_WEIGHT_CHOICE,
                 device: str=DEFAULT_DEVICE, 
                 **kwargs) -> None:
        
        learner_cls = _APICVAEComplex4MSO_LearnerCVAEComplex
        self.learner = learner_cls.init_inference_from_saved_dn(
            saved_dn, weight_choice, device, **kwargs
        )

    def init_smis(self, item: ComplexAAItem, 
                  num_samples: int, batch_size: int=8,
                  validate_mol: bool=True, unique_mol: bool=True,
                  unique_valid_mol: bool=True) -> List[str]:
        sampler = self.learner.sample(item, batch_size)
        init_smi_lst = []
        valid_smi_set = set()
        pbar = tqdm(total=num_samples, desc="Init SMILES")
        num_attempts = 0
        while len(init_smi_lst) < num_samples:
            num_attempts += 1
            prd_seq, c_attn = next(sampler)
            smi = self.learner._seq_to_smi(prd_seq[:-1])
            is_unique = smi not in init_smi_lst
            is_valid = not smi.startswith('!')
            if is_valid:
                is_valid_unique = smi.split('\t')[0] not in valid_smi_set
                valid_smi_set.add(smi.split('\t')[0])
            else:
                is_valid_unique = False
            if all((
                is_unique or not unique_mol,
                is_valid or not validate_mol,
                is_valid_unique or not unique_valid_mol
            )):
                init_smi_lst.append(smi)
                pbar.update()
            succ = len(init_smi_lst) / num_attempts * 100
            pbar.set_postfix({"succ": f"{succ:.3f}%"})
        pbar.close()
        succ = len(init_smi_lst) / num_attempts * 100
        return init_smi_lst, succ

class _APICVAEComplex4MSO_LearnerCVAEComplex(LearnerCVAEComplex):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._seq_to_smi_cache = bidict()
        self.c_state: Tensor = None

    @torch.no_grad()
    def init_c_state(self, item: ComplexAAItem) -> Tensor:
        """Initialize Fixed protein embeding `c_state`
        """
        self.model.eval()
        self.esm_model.eval()
        batch = self.collate_fn([item])
        assert batch.c is not None, "How can I predict without protein?"
        c_repr = self._esm_inference_batch(batch).to(self.device)
        c_state, c_attn = self.model.encode_c(c_repr, batch.c_len)
        # c_state = [1, c_emb_dim]
        self.c_state = c_state
        # self.c_state = torch.zeros_like(c_state)
        return c_state

    def _seq_to_smi(self, seq: List[int]) -> str:
        """seq -> smi
        [3, 4] -> COCCC\t!3,4
        [5, 6] -> !5,6
        """
        seq = tuple(seq)
        if seq in self._seq_to_smi_cache:
            smi = self._seq_to_smi_cache[seq]
        else:
            src = f"!{','.join(str(x) for x in seq)}"
            try:
                assert len(seq) > 0, "Empty seq"
                frags_lst = self.x_vocab.itos(seq)
                smi = mol_to_frag.detokenize(frags_lst)
                assert len(smi) > 0, "Empty smi"
                smi = f"{smi}\t{src}"
            except Exception as err:
                # warnings.warn(f"{smi} -> {repr(err)}")
                # Illegal sequences can also be restored through SMILES (compatible with MSO)
                smi = src
            self._seq_to_smi_cache[seq] = smi
        return smi

    def _smi_to_seq(self, smi: str) -> List[int]:
        """smi -> seq
        COCCC\t!3,4 -> [3, 4]
        *Reverse with cache _seq_to_smi()*
        """
        if smi in self._seq_to_smi_cache.inverse:
            seq = self._seq_to_smi_cache.inverse[smi]
            return list(seq)
        else:
            raise ValueError(f"For {smi}, seq -> smi must be performed first")

    @torch.no_grad()
    def _smis_to_embs(self, smis: List[str]) -> Tensor:
        self.model.eval()
        bs = len(smis)
        item_lst = []
        for smi in smis:
            seq = self._smi_to_seq(smi)
            item = ComplexAAItem(
                id=smi,
                x=[self.x_vocab.special_idx.sos, *seq[:self.x_max_len - 2], self.x_vocab.special_idx.eos]
            )
            item_lst.append(item)
        batch = self.collate_fn(item_lst)
        c_state = self.c_state.repeat(bs, 1)
        x_state = self.model.encode_x(batch.x, batch.x_len)
        xc_state = torch.cat((x_state, c_state), dim=1)
        recog_mu, recog_log_var = self.model.recog_head(xc_state)
        z = recog_mu
        # z = self.model.reparametrize(recog_mu, recog_log_var)
        return z

    @torch.no_grad()
    def _embs_to_smis(self, embs: Tensor) -> List[str]:
        self.model.eval()
        z = embs
        bs = z.shape[0]
        c_state = self.c_state.repeat(bs, 1)
        z = torch.cat((z, c_state), dim=1)
        recon, prd, prd_len = self.model.decode(z, self.x_vocab.special_idx, self.x_max_len)
        smis = []
        for i in range(bs):
            prd_seq = prd[i][:prd_len[i]].tolist()
            smis.append(self._seq_to_smi(prd_seq[:-1]))
        return smis

    # Output: seq means smi
    def seq_to_emb(self, seq: List[str]):
        """Helper function to calculate the embedding (molecular descriptor) for input sequnce(s)

        Args:
            seq: Single sequnces or list of sequnces to encode.
        Returns:
            Embedding of the input sequnce(s).
        """
        isnot_batch = isinstance(seq, str)
        smis = [seq] if isnot_batch else list(seq)
        assert isinstance(smis, list) and all(isinstance(smi, str) for smi in smis)
        embs = self._smis_to_embs(smis)
        embs = embs.cpu().detach().numpy()
        embedding = embs[0] if isnot_batch else embs
        return embedding

    # Output: seq means smi
    def emb_to_seq(self, embedding: np.ndarray):
        """Helper function to calculate the sequnce(s) for one or multiple (concatinated)
        embedding.

        Args:
            embedding: array with n_samples x num_features.
        Returns:
            sequnce(s).
        """
        isnot_batch = embedding.ndim == 1
        embs = np.expand_dims(embedding, 0) if isnot_batch else embedding
        assert embs.ndim == 2
        embs = torch.from_numpy(embs).to(self.device)
        smis = self._embs_to_smis(embs)
        seq = smis[0] if isnot_batch else smis
        return seq

_TSMILES = NewType('_TSMILES', str)
_TFitness = Dict[_TSMILES, Dict[str, Number]]
_TPopu = Set[_TSMILES]

class APICVAEComplex4Population(APICVAEComplex):
    """High level API for Population
    """

    TSMILES = _TSMILES
    TFitness= _TFitness
    TPopu = _TPopu

    def init_popu(self,
                  c_item: ComplexAAItem, fitness_fn: Callable[[_TPopu], _TFitness],
                  popu_size: int=200, out_rate: float=0.1, batch_size: int=8,
                  random_seed: int=20230403) -> None:
        """Initialize Population
        """

        self.c_item = c_item
        self.fitness_fn = fitness_fn
        # Return dict: {smi: {tox: 12, qed:1, scaled: 0.7}}
        # For scaled: The larger, the better 
        self.popu_size = popu_size
        self.batch_size = batch_size
        self.out_size = round(popu_size * out_rate)
        assert 1 < self.out_size < self.popu_size, \
            (f"The out size is {self.out_size}, which must be "
             f"greater than 1 and smaller than the population size")
        self.r = random.Random(random_seed)

        self.popu_set: _TPopu = set()
        self.seq_dic: Dict[_TSMILES, List[int]] = dict()
        self.popu_set_history: Dict[str, _TPopu] = dict()

    def generate_popu(self, desc="Initialize Population") -> int:
        sampler = self.chem_sample(self.c_item, self.batch_size)
        sample_res_gen = islice(sampler, self.popu_size)
        pbar = tqdm(sample_res_gen, total=self.popu_size, desc=desc)
        for res in pbar:
            res = edict(res)
            self.popu_set.add(res.smi)
            self.seq_dic[res.smi] = res.seq
        assert len(self.popu_set) == self.popu_size
        return sampler.num_attempts

    def evaluate_popu(self, popu_set: _TPopu=None) -> Tuple[_TSMILES, _TFitness]:
        popu_set = ifn(popu_set, self.popu_set)
        fitness_dic = self.fitness_fn(popu_set)
        assert set(fitness_dic.keys()) == popu_set
        sorted_lst = sorted(fitness_dic, key=lambda k: fitness_dic[k]["scaled"])
        return sorted_lst, fitness_dic

    def update_popu(self, desc="Update Population") -> int:
        sorted_lst, fitness_dic = self.evaluate_popu()
        self.popu_set = set(sorted_lst[self.out_size:])
        assert len(self.popu_set) == self.popu_size - self.out_size
        items = self.random_popu_items()
        optimizer = self.chem_optimize(items, self.batch_size)
        childs = optimizer.ordered_iter()
        pbar = tqdm(total=self.popu_size, initial=len(self.popu_set), desc=desc)
        while len(self.popu_set) < self.popu_size:
            cid, res = next(childs)
            if res.smi not in self.popu_set:
                self.popu_set.add(res.smi)
                self.seq_dic[res.smi] = res.seq
                pbar.update()
        pbar.close()
        assert len(self.popu_set) == self.popu_size
        return optimizer.num_attempts

    def log_popu(self, stage: str) -> str:
        popu_set = self.popu_set.copy()
        self.popu_set_history[stage] = popu_set
        sorted_lst, fitness_dic = self.evaluate_popu(popu_set)
        score_lst_dic = defaultdict(list)
        for score_dic in fitness_dic.values():
            for k, v in score_dic.items():
                score_lst_dic[k].append(v)
        res_dic = {k: np.mean(v) for k, v in score_lst_dic.items()}
        res_str = f"{stage} -> {pretty_kv(res_dic, ndigits=5)}"
        return res_str

    def top_popu(self, top_k: int=50) -> _TFitness:
        sorted_lst, fitness_dic = self.evaluate_popu(self.popu_set)
        top_lst = sorted_lst[:-top_k:-1]
        res_lst = [{"smi": smi, **fitness_dic[smi]} for smi in top_lst]
        return res_lst

    def run(self, epochs: int, top_k: int=50) -> List[_TFitness]:
        num_attempts = self.generate_popu()
        log_res = self.log_popu(f"Init     ")
        logging.info(f"{log_res}, succ: {self.popu_size/num_attempts*100:.3f}%")
        for epoch in range(epochs):
            num_attempts = self.update_popu(desc=f"Update Population {epoch:>3}")
            log_res = self.log_popu(f"Epoch.{epoch:<3}")
            logging.info(f"{log_res}, succ: {self.out_size/num_attempts*100:.3f}%")
        res_lst = self.top_popu(top_k)
        return res_lst

    def random_popu_items(self):
        popu_lst = list(self.popu_set)
        while True:
            idx_lst = self.r.sample(range(len(popu_lst)), k=len(popu_lst))
            for idx in idx_lst:
                smi = popu_lst[idx]
                item = ComplexAAItem(
                    id=idx,
                    x=[self.learner.x_vocab.special_idx.sos, 
                    *self.seq_dic[smi][:self.learner.x_max_len - 2], 
                    self.learner.x_vocab.special_idx.eos],
                    c=self.c_item.c)
                yield item

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
        self.log_timestamp: Dict[int, float] = dict()

    def generate_popu(self, desc="Initialize Population") -> int:
        sampler = self.chem_sample(self.c_item, self.batch_size)
        sample_res_gen = islice(sampler, self.popu_size)
        pbar = tqdm(sample_res_gen, total=self.popu_size, desc=desc)
        self.fill_popu(pbar)
        return sampler.num_attempts

    def fill_popu(self, res_lst: list) -> None:
        for res in res_lst:
            res = edict(res)
            self.popu_lst.append(res.smi)
            self.seq_dic[res.smi] = res.seq
        assert len(self.popu_lst) == self.popu_size
        return len(res_lst)

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
        self.log_timestamp[epoch] = time.time()
        
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

    def run(self, epochs: int, top_k: int=-1, fill_popu_res_lst: list[dict]=None) -> List[_TFitness]:
        # T0*(D^k) = Tf
        self.Tf = self.T0 * (self.D ** (epochs - 1))
        if fill_popu_res_lst is None:
            num_attempts = self.generate_popu()
        else:
            num_attempts = self.fill_popu(fill_popu_res_lst)
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
        dic['timestamp'] = [self.log_timestamp[i] for i in range(len(self.log_timestamp))]
        return dic

_TEmb = Tuple[float]
_TPopuGP = List[_TEmb]
class APICVAEComplex4GP(APICVAEComplex):
    """High level API for GP

    - https://github.com/mkusner/grammarVAE
    """

    TSMILES = _TSMILES
    TFitness= _TFitness
    TEmb = _TEmb
    TPopuGP = _TPopuGP
    TPopuSA = _TPopuSA

    def __init__(self, 
                 saved_dn: StrPath=DEFAULT_SAVED_DN, weight_choice: str=DEFAULT_WEIGHT_CHOICE,
                 device: str=DEFAULT_DEVICE, 
                 **kwargs) -> None:
        super().__init__(saved_dn, weight_choice, device, **kwargs)

        global gp
        from ..utils import gaussian_process as gp

    def init_popu(self,
                  c_item: ComplexAAItem, fitness_fn: Callable[[_TPopuSA], _TFitness],
                  popu_size: int=200, num_candidates: int = 50, num_restarts: int = 10, raw_samples: int = 256,
                  latent_bounds: float | tuple=None,
                  batch_size: int=8, random_seed: int=20230403) -> None:
        """Initialize Population
        """

        self.c_item = c_item
        self.fitness_fn = fitness_fn
        # Return dict: {smi: {tox: 12, qed:1, scaled: 0.7}}
        # For scaled: The larger, the better 
        self.popu_size = popu_size
        self.batch_size = batch_size

        self.num_candidates = num_candidates
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        if isinstance(latent_bounds, Number):
            self.latent_bounds = (-latent_bounds, latent_bounds)
        else:
            self.latent_bounds = latent_bounds

        self.r = np.random.default_rng(random_seed)
        self.gp_state_dict = None
        self.gp_bounds = None

        self.popu_lst: _TPopuGP = list()
        self.emb_dic: Dict[_TEmb, _TSMILES] = dict()
        self.trans_smi_lst: Dict[int, _TPopuSA] = dict()
        self.log_timestamp: Dict[int, float] = dict()

    def generate_popu(self, desc="Initialize Population") -> int:
        sampler = self.chem_sample(self.c_item, self.batch_size)
        sample_res_gen = islice(sampler, self.popu_size)
        pbar = tqdm(sample_res_gen, total=self.popu_size, desc=desc)
        res_lst = list(pbar)
        self.fill_popu(res_lst)
        return res_lst

    def fill_popu(self, res_lst: list) -> None:
        emb_lst = self.seqs_to_embs([res["seq"] for res in res_lst])
        for res, emb in zip(res_lst, emb_lst):
            res = edict(res)
            self.popu_lst.append(emb)
            self.emb_dic[emb] = res.smi
        embs = torch.tensor(emb_lst, dtype=float)
        if self.latent_bounds is None:
            self.gp_bounds = torch.stack([embs.min(0).values, embs.max(0).values])
        else:
            self.gp_bounds = torch.tensor(self.latent_bounds, dtype=float)[:, None].repeat(1, embs.shape[1])
        assert len(self.popu_lst) == self.popu_size
        return res_lst

    def evaluate_popu(self, popu_lst: _TPopuGP=None) -> Tuple[_TSMILES, _TFitness]:
        popu_lst = ifn(popu_lst, self.popu_lst)
        if isinstance(popu_lst[0], str):
            smi_lst = popu_lst
        else:
            smi_lst = [self.emb_dic[emb] for emb in self.popu_lst]
        fitness_dic = self.fitness_fn(smi_lst)
        sorted_lst = sorted(fitness_dic, key=lambda k: fitness_dic[k]["scaled"], reverse=True)
        return sorted_lst, fitness_dic

    def update_popu(self, desc="Update Population") -> Tuple[int, int]:
        smi_lst = [self.emb_dic[emb] for emb in self.popu_lst]
        fitness_dic = self.fitness_fn(smi_lst)

        train_x = torch.tensor(self.popu_lst, dtype=float)
        train_obj = torch.tensor([fitness_dic[smi]["scaled"] for smi in smi_lst], dtype=float)[:, None]

        # https://botorch.org/tutorials/vae_mnist
        model, candidates = gp.fitted_model_optimize_acqf(
            train_x=train_x,
            train_obj=train_obj,
            state_dict=self.gp_state_dict,
            q=self.num_candidates,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            bounds=self.gp_bounds,
        )
        self.gp_state_dict = model.state_dict()

        candidates = [tuple(emb) for emb in candidates.cpu().tolist()]
        seq_lst = self.embs_to_seqs(candidates)
        res_lst = []
        for idx, (seq, emb) in enumerate(zip(seq_lst, candidates)):
            res = edict(seq=seq, smi=None, frags=None, idx=idx)
            try:
                assert len(res.seq) > 0, "Empty seq"
                res.frags = self.learner.x_vocab.itos(res.seq)
                res.smi = mol_to_frag.detokenize(res.frags)
                assert len(res.smi) > 0, "Empty smi"
                assert embed_smi_cached(res.smi), "Not embedable smi"
            except Exception as err:
                # print(f"{res} -> {repr(err)}")
                pass
            else:
                res_lst.append(res)

        for res in res_lst:
            self.popu_lst.append(candidates[res.idx])
            self.emb_dic[candidates[res.idx]] = res.smi
        return res_lst

    def log_popu(self, epoch: int, res_lst: list[dict]) -> str:
        trans_smi_lst = [res["smi"] for res in res_lst]
        self.trans_smi_lst[epoch] = trans_smi_lst
        self.log_timestamp[epoch] = time.time()

        popu_lst = self.popu_lst.copy()
        # sorted_lst, fitness_dic = self.evaluate_popu(popu_lst)
        sorted_lst, fitness_dic = self.evaluate_popu(trans_smi_lst)
        score_lst_dic = defaultdict(list)
        for score_dic in fitness_dic.values():
            for k, v in score_dic.items():
                score_lst_dic[k].append(v)
        res_dic = {k: np.mean(v) for k, v in score_lst_dic.items()}
        res_dic["uniqueness"] = len(set(popu_lst)) / len(popu_lst)
        res_str = f"Epoch.{epoch:<3} -> {pretty_kv(res_dic, ndigits=5)}"
        return res_str

    def top_popu(self, top_k: int=-1) -> _TFitness:
        popu_lst = self.popu_lst.copy()
        sorted_lst, fitness_dic = self.evaluate_popu(popu_lst)
        top_lst = sorted_lst[:top_k]
        res_lst = [{"smi": smi, **fitness_dic[smi]} for smi in top_lst]
        return res_lst

    def run(self, epochs: int, top_k: int=-1, fill_popu_res_lst: list[dict]=None) -> List[_TFitness]:
        if fill_popu_res_lst is None:
            res_lst = self.generate_popu()
        else:
            res_lst = self.fill_popu(fill_popu_res_lst)
        log_res = self.log_popu(0, res_lst)
        logging.info(f"{log_res}, succ: {len(res_lst)/self.popu_size*100:.3f}%")

        for epoch in range(epochs):
            res_lst = self.update_popu()
            log_res = self.log_popu(epoch+1, res_lst)
            logging.info(f"{log_res}, succ: {len(res_lst)/self.num_candidates*100:.3f}%, trans: {len(res_lst)}")
        res_lst = self.top_popu(top_k)
        return res_lst

    def seqs_to_embs(self, seqs: list[list[int]]) -> list[tuple[float]]:
        embs = []
        for chunk in chunked(list(enumerate(seqs)), self.batch_size):
            batch = self.learner.collate_fn([
                ComplexAAItem(
                    id=f"{int(time.time()*1000)}_{idx}",
                    x=[self.learner.x_vocab.special_idx.sos, 
                       *seq[:self.learner.x_max_len - 2], 
                       self.learner.x_vocab.special_idx.eos],
                    c=self.c_item.c) for idx, seq in chunk
                ])
            z = self.learner.encode_emb(batch)
            embs.extend(z.cpu().tolist())
        embs = [tuple(emb) for emb in embs]
        return embs

    def embs_to_seqs(self, embs: list[list[float]]) -> list[list[int]]:
        seqs = []
        for chunk in chunked(list(enumerate(embs)), self.batch_size):
            batch = self.learner.collate_fn([self.c_item] * len(chunk))
            z = torch.tensor([emb for idx, emb in chunk], dtype=torch.float)
            seq_lst = self.learner.decode_emb(z, batch)
            seqs.extend(seq_lst)
        seqs = [tuple(seq[:-1]) for seq in seqs]
        return seqs

    def info(self) -> Dict:
        dic = dict()
        dic['popu'] = [self.trans_smi_lst[i] for i in range(len(self.trans_smi_lst))]
        _, dic['fitness'] = self.evaluate_popu(list(set(chain.from_iterable(dic['popu']))))
        dic['para'] = dict(num_candidates=self.num_candidates, 
                           num_restarts=self.num_restarts, 
                           raw_samples=self.raw_samples)
        dic['timestamp'] = [self.log_timestamp[i] for i in range(len(self.log_timestamp))]
        return dic

ITEM_MAKER_INPUT_TYPES = {'seq', 'pdb', 'url', 'pdb_fn', 'pdb_id'}
ITEM_MAKER_GROUNDTRUTH_LIGAND_TYPES = {'residue_name_in_pdb', 'pdb_fn', 'sdf_fn', 'sdf', 'sdf_url'}
class ItemMaker:
    def __init__(self, input_data, input_type, 
                 input_chain_id: str=None,
                 groundtruth_ligand_data: str=None,
                 groundtruth_ligand_type: str=None) -> None:
        self.d = defaultdict(lambda: None)
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

        if groundtruth_ligand_data is not None:
            if groundtruth_ligand_type == 'sdf_fn':
                ligand_sdf_fn = self._ligand_from_sdf_fn(groundtruth_ligand_data)
            elif groundtruth_ligand_type == 'sdf_url':
                ligand_sdf_fn = self._ligand_from_url(groundtruth_ligand_data)
            elif groundtruth_ligand_type == 'sdf':
                ligand_sdf_fn = self._ligand_from_sdf(groundtruth_ligand_data)
            else:
                raise NotImplementedError(f"Unknow {groundtruth_ligand_type=}")
        else:
            ligand_sdf_fn = None

        input_kwargs = dict(input_chain_id=input_chain_id, ligand_sdf_fn=ligand_sdf_fn)
        if input_type == 'seq':
            self.item = self._from_seq(input_data, **input_kwargs)
        elif input_type == 'pdb':
            self.item = self._from_pdb(input_data, **input_kwargs)
        elif input_type == 'url':
            self.item = self._from_url(input_data, **input_kwargs)
        elif input_type == 'pdb_fn':
            self.item = self._from_pdb_fn(input_data, **input_kwargs)
        elif input_type == 'pdb_id':
            self.item = self._from_pdb_id(input_data, **input_kwargs)
        else:
            raise Exception(f'Unknow input type: {input_type}')

    def _ligand_from_sdf_fn(self, sdf_fn: StrPath) -> Path:
        sdf_fn = use_path(file_path=sdf_fn, new=False)
        self.d['ligand_sdf_fn'] = sdf_fn
        sdf = sdf_fn.read_bytes()
        return self._ligand_from_sdf(sdf)

    def _ligand_from_sdf(self, sdf: bytes) -> Path:
        assert isinstance(sdf, bytes)
        self.d['ligand_sdf'] = sdf
        fn = Path(self.tmpdir) / "groundtruth_lig.sdf"
        fn.write_bytes(sdf)
        return fn

    def _ligand_from_url(self, url: str) -> Path:
        assert isinstance(url, str)
        self.d['ligand_url'] = url
        sdf = bytes(download_pbar(url))
        return self._ligand_from_sdf(sdf)

    def _from_seq(self, seq: str, **kwargs):
        assert isinstance(seq, str)
        item_id = str(int(time.time()*1000))
        return ComplexAAItem(id=item_id, x=[], c=seq)

    def _from_aa(self, aa: complex_to_aa.ComplexAAExtract):
        assert isinstance(aa, complex_to_aa.ComplexAAExtract)
        self.d['aa'] = aa
        item_id = str(int(time.time()*1000))
        return ComplexAAItem(
            id=item_id, x=[], c=aa.seq, 
            c_rel=norm_rel_fn(aa.pocket_dist) if aa.pocket_dist is not None else None
        )

    def _from_pdb(self, pdb: bytes, input_chain_id: str, ligand_sdf_fn: StrPath):
        assert isinstance(pdb, bytes)
        self.d['pdb'] = pdb
        fn = Path(self.tmpdir) / "rec.pdb"
        fn.write_bytes(pdb)
        aa = complex_to_aa.extract(protein_pdb_fn=fn, 
                                    ligand_sdf_fn=ligand_sdf_fn, 
                                    chain_id=input_chain_id,
                                    logging_handle=logging.warning)
        return self._from_aa(aa)

    def _from_pdb_fn(self, pdb_fn: StrPath, **kwargs):
        pdb_fn = use_path(file_path=pdb_fn, new=False)
        self.d['pdb_fn'] = pdb_fn
        pdb = pdb_fn.read_bytes()
        return self._from_pdb(pdb, **kwargs)

    def _from_url(self, url: str, **kwargs):
        assert isinstance(url, str)
        self.d['url'] = url
        pdb = bytes(download_pbar(url))
        return self._from_pdb(pdb, **kwargs)

    def _from_pdb_id(self, pdb_id: str, **kwargs):
        assert isinstance(pdb_id, str)
        self.d['pdb_id'] = pdb_id
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        return self._from_url(url, **kwargs)
    
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
