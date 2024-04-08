from collections import Counter
from pathlib import Path
import pickle
import shutil
from typing import Dict, List
import torch
import logging
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from .raw_third_dataset import RawThirdDataset
from ..utils import StrPath, Toc, get_file_size, \
    pretty_kv, auto_dump, summary_arr, complex_to_aa

class CrossDockedDataset(RawThirdDataset):
    r"""CrossDocked dataset with
    
    - Protein: Amino acid sequence, center of mass for each amino acid.
    - Ligand: Fragment, word.

    Source:

    Structure for `meta`

    ```json
    {
        "id": "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0"
        "protein": "1B57_HUMAN_25_300_0/5u98_D_rec.pdb", 
        "ligand": "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0.sdf", 
        "pocket": "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0_pocket10.pdb", 
        "rmsd": 0.367042, 
        "split": "train"                        // "test" | "other"
        "smi": "CCCCCCCCCC..."
    }
    ```

    Structure for `aa`

    ```json
    {
        "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0": {
            "id": "A", 
            "len": 123,                         // len(seq)
            "seq": "AMGQSTSN...",               // length = L
            "ids": [-1, 1, 2, "<...>"],         // length = L
            "pos": "<np.ndarray>",              // shape = [L, 3]
            "pocket_cent": "<np.ndarray>",      // shape = [3]
            "pocket_dist": "<np.ndarray>"       // shape = [L]
        },
        '<id>': {},
        ...
    }
    ```
    """

    class _LostException(Exception):
        pass

    def __init__(self, cache_dn: StrPath):
        super().__init__()
        _cache_dn = Path(cache_dn)
        self.fn_dic = {
            "meta": _cache_dn / "meta.json",
            "complex_to_aa": _cache_dn / "complex_to_aa.msgpack",
            "mol_to_frag": _cache_dn / "mol_to_frag.json",
            "frag_to_word": _cache_dn / "frag_to_word.json",
            "frag_vocab": _cache_dn / "frag_vocab.json",
            "word_vocab": _cache_dn / "word_vocab.json"
        }

    def _preprocess_meta(self, sbdd_dir: Path, crossdocked_dir: Path=None, 
                        is_dev=False, only_known=True) -> List[Dict]:

        with open(sbdd_dir / "split_by_name.pt", "rb") as f:
            split_index = torch.load(f)
        with open(sbdd_dir / "index.pkl", "rb") as f:
            filter_index = pickle.load(f)

        _lst = []
        for k, v in split_index.items():
            for x in v:
                _lst.append((x[1], k))
        split_dic = dict(_lst)

        if only_known:
            filter_index = [x for x in filter_index if x[1] in split_dic]
        if is_dev:
            filter_index = filter_index[:1000]

        meta_lst = []
        pbar = tqdm(filter_index, desc=f"Preprocess [meta]")
        status = edict(dict(found=0, copied=0, lost=0, bad=0))
        for x in pbar:
            try:
                _dic = edict(dict(
                    id = str(Path(x[1]).with_suffix('').as_posix()),
                    protein = x[2], ligand = x[1], pocket = x[0], 
                    rmsd = x[3], split = split_dic.get(x[1], "other"),
                    smi = None
                ))
                _cond = [
                    _dic.ligand and (sbdd_dir / _dic.ligand).exists(),                             # 0
                    _dic.pocket and (sbdd_dir / _dic.pocket).exists(),                             # 1
                    _dic.protein and (sbdd_dir / _dic.protein).exists(),                           # 2
                    _dic.protein and crossdocked_dir and (crossdocked_dir / _dic.protein).exists() # 3
                ]

                if not _cond[0]:
                    raise self._LostException(f"Lost ligand: {_dic.ligand}")
                if not _cond[1]:
                    raise self._LostException(f"Lost pocket: {_dic.pocket}")
                if not (_cond[2] or _cond[3]):
                    raise self._LostException(f"Lost protein: {_dic.protein}")

                try:
                    _dic.smi = complex_to_aa.sdf_to_smi(sbdd_dir / _dic.ligand)
                    assert _dic.smi, "Empty SMILES from SDF"
                except Exception as error:
                    raise Exception(f"Bad ligand: {_dic.ligand}\n{error}")

                if (not _cond[2]) and _cond[3]:
                    # Copy protein file if it exist in full CrossDocked dataset
                    shutil.copyfile(crossdocked_dir / _dic.protein, sbdd_dir / _dic.protein)
                    status.copied += 1

            except Exception as error:
                logging.error(error)
                if isinstance(error, self._LostException):
                    status.lost += 1
                else:
                    status.bad += 1
            else:
                meta_lst.append(_dic)
                status.found += 1

            pbar.set_postfix(status)

        logging.info(f"status -> {pretty_kv(status)}")

        # Check unique meta.id
        id_counter = Counter(meta.id for meta in meta_lst)
        id_dup_dic = {id: cnt for id, cnt in id_counter.items() if cnt > 1}
        if len(id_dup_dic) > 0:
            logging.warning(f"Meta id is not unique!\n{pretty_kv(id_dup_dic)}")

        return meta_lst

    def preprocess(self, sbdd_dir: Path, crossdocked_dir: Path=None,
                    n_jobs=4, is_dev=False, 
                    frag_min_freq=1, word_min_freq=1):
        toc = Toc()

        # Preprocess meta
        name = "meta"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            meta_lst = self._preprocess_meta(sbdd_dir, crossdocked_dir, is_dev)
            logging.info(f"Done! toc: {toc():.3f}s")
            auto_dump(meta_lst, self.fn_dic[name], json_indent=True)
        else:
            logging.info(f"Skip {name}!")
            skip = True
            meta_lst = self.source(name)

        logging.info(f"{'Loaded' if skip else 'Saved'}! "
                     f"{name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
        if "split" in meta_lst[0]:
            logging.info(f'split -> {pretty_kv(dict(Counter(x["split"] for x in meta_lst)))}')
        logging.info(f"{summary_arr(meta_lst, key=lambda x: len(x['smi']))}")

        # Preprocess complex_to_aa
        name = "complex_to_aa"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            complex_to_aa_dic = self._preprocess_complex_to_aa(sbdd_dir, meta_lst, n_jobs=n_jobs)
            logging.info(f"Done! toc: {toc():.3f}s")
            auto_dump(complex_to_aa_dic, self.fn_dic[name])
        else:
            logging.info(f"Skip {name}!")
            skip = True
            complex_to_aa_dic = self.source(name)
            
        logging.info(f"{'Loaded' if skip else 'Saved'}! "
                     f"{name} -> {self.fn_dic['complex_to_aa']}, toc: {toc():.3f}s, "
                     f"size: {get_file_size(self.fn_dic['complex_to_aa'], pretty=True)}")
        logging.info(f"{summary_arr(complex_to_aa_dic.values(), key=lambda x: len(x.seq))}")

        # Preprocess common flow
        self._preprocess_common_flow(meta_lst, n_jobs, frag_min_freq, word_min_freq)
