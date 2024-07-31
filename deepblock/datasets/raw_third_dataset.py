from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union
import warnings
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import logging
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from ..utils import Toc, Vocab, TqdmTrackerJoblibCallback, \
    mol_to_frag, smi_to_word, auto_dump, auto_load, \
    time_limit, summary_arr, complex_to_aa, hook_joblib

class RawThirdDataset(Dataset):
    """Base dataset class for third-party raw data.
    """

    @staticmethod
    def convert_ndarray(obj: Union[List, None]) -> Union[np.ndarray, None]:
        return np.array(obj) if obj else None

    def __init__(self):
        self.fn_dic: Dict[Union[Tuple[str, ...], str], Path] = {}
        self.mem: Dict[Union[Tuple[str, ...], str], object] = {}

    def _source_meta(self, obj: List[Dict]):
        return list(map(edict, obj))

    def _source_vocab(self, obj: Dict):
        return Vocab(**obj)

    _source_frag_vocab = _source_vocab
    _source_word_vocab = _source_vocab

    def source(self, name: str, keep: bool=True):
        """Restore objects from local files with hook support.
        """
        if name in self.mem:
            obj = self.mem[name]
        else:
            hook = getattr(self, f'_source_{name}', None)
            obj = auto_load(self.fn_dic[name])
            if hook is not None:
                obj = hook(obj)
            if keep:
                self.mem[name] = obj
        return obj

    def _preprocess_meta(self, *args, **kwargs) -> List[Dict]:
        raise Exception("Not defined")

    def _preprocess_mol_to_frag(self, mol_set_lst, n_jobs):
        def _par_f1(smi: str):
            try:
                with time_limit(5):
                    toc = Toc()
                    frag_seq, act_smi = mol_to_frag.tokenize(
                        smi, return_act_smi=True)
                    result = dict(frag_seq=frag_seq, act_smi=act_smi, toc=toc())
            except Exception as err:
                logging.error(f"{smi} -> {err}")
                result = err
            return result

        pbar = tqdm(total=len(mol_set_lst), desc=f'[preprocess] mol_to_frag (parallel)')
        joblib_cb = TqdmTrackerJoblibCallback(pbar)
        with hook_joblib(joblib_cb):
            result_lst = Parallel(n_jobs=n_jobs)(
                delayed(_par_f1)(x) for x in mol_set_lst)
        pbar.close()

        _dic = {mol: result for mol, result in zip(
            mol_set_lst, result_lst) if not isinstance(result, Exception)}
        return _dic

    def _preprocess_frag_to_word(self, frag_set_lst):
        _lst = []
        for frag in frag_set_lst:
            try:
                word_seq = smi_to_word.tokenize(frag)
                _lst.append((frag, word_seq))
            except Exception as err:
                logging.error(f"{frag} -> {err}")
        return dict(_lst)

    def _preprocess_complex_to_aa(self, base_dir: Path, 
                                  meta_lst: List[Dict], n_jobs: int=1, 
                                  ignore_warning: bool=False) -> Dict[str, complex_to_aa.ComplexAAExtract]:

        def _par_f1(meta: Dict):
            try:
                if ignore_warning:
                    warnings.filterwarnings('ignore', module=complex_to_aa.__name__)
                result = complex_to_aa.extract(base_dir / meta.protein, base_dir / meta.ligand,
                                               r"{chain_close_pocket}")
            except Exception as err:
                logging.error(f"{meta} -> \n{err}")
                result = err
            return result

        pbar = tqdm(total=len(meta_lst), desc=f'[preprocess] complex_to_aa (parallel)')
        joblib_cb = TqdmTrackerJoblibCallback(pbar)
        with hook_joblib(joblib_cb):
            result_lst = Parallel(n_jobs=n_jobs)(
                delayed(_par_f1)(meta) for meta in meta_lst)
        pbar.close()
  
        _dic = {meta.id: result for meta, result in zip(
            meta_lst, result_lst) if not isinstance(result, Exception)}

        return _dic

    def _source_complex_to_aa(self, obj) -> Dict[str, complex_to_aa.ComplexAAExtract]:
        _lst = []
        for k, v in obj.items():
            _v = complex_to_aa.ComplexAAExtract(**v)
            _v.pos = self.convert_ndarray(_v.pos)
            _v.pocket_cent = self.convert_ndarray(_v.pocket_cent)
            _v.pocket_dist = self.convert_ndarray(_v.pocket_dist)
            _lst.append((k, _v))
        return dict(_lst)

    def _preprocess_common_flow(self, meta_lst, n_jobs, frag_min_freq, word_min_freq):
        toc = Toc()

        # Intermediate processing mol
        mol_tup = tuple(x["smi"] for x in meta_lst)
        mol_set_lst = sorted(set(mol_tup))

        # Preprocess mol_to_frag
        name = "mol_to_frag"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            mol_to_frag_dic = self._preprocess_mol_to_frag(mol_set_lst, n_jobs)
            logging.info(f"Done! toc: {toc():.3f}s")
            auto_dump(mol_to_frag_dic, self.fn_dic[name])
        else:
            logging.info(f"Skip {name}!")
            skip = True
            mol_to_frag_dic = self.source(name)

        logging.info(f"{'Loaded' if skip else 'Saved'}! "
                     f"{name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
        logging.info(f"{summary_arr(mol_to_frag_dic.values(), key=lambda x: len(x['frag_seq']))}")

        # Intermediate processing frag
        frag_seqs = list(x['frag_seq'] for x in mol_to_frag_dic.values())
        frag_tup = tuple(chain.from_iterable(frag_seqs))
        frag_set_lst = sorted(set(frag_tup))

        # Preprocess frag_to_word
        name = "frag_to_word"
        toc()
        frag_to_word_dic = self._preprocess_frag_to_word(frag_set_lst)
        auto_dump(frag_to_word_dic, self.fn_dic[name])
        logging.info(f"Saved! {name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
        logging.info(f"{summary_arr(frag_to_word_dic.values(), key=len)}")

        # Intermediate processing word
        word_seqs = list(frag_to_word_dic.values())
        word_tup = tuple(chain.from_iterable(word_seqs))

        # frag_vocab
        jobs = {
            "frag_vocab": (frag_tup, frag_seqs, frag_min_freq),
            "word_vocab": (word_tup, word_seqs, word_min_freq),
        }
        for name, (_tup, _seqs, _min_freq) in jobs.items():
            toc()
            _vocab = Vocab.from_seq(_tup, min_freq=_min_freq)
            _vocab_dic = _vocab.to_dict()
            auto_dump(_vocab_dic, self.fn_dic[name])
            logging.info(f"Saved! {name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
            num_coverage = _vocab.coverage(_seqs)
            logging.info(f"coverage -> {num_coverage}/{len(_seqs)}={num_coverage/len(_seqs)*100:.3f}%")
            logging.info(f"{summary_arr(_vocab, key=len)}")

    def preprocess(self, *args, **kwargs):
        raise Exception("Not defined")

    def __len__(self):
        return len(self.source('meta'))

    def __getitem__(self, idx: int):
        return self.source('meta')[idx]
