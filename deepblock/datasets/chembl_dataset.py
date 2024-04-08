import csv
from itertools import islice
from pathlib import Path
import logging
from easydict import EasyDict as edict

from .raw_third_dataset import RawThirdDataset
from ..utils import StrPath, Toc, \
    auto_dump, unique_by_key, summary_arr

class ChEMBLDataset(RawThirdDataset):
    """ChEMBL dataset with fragment and word."""

    def __init__(self, cache_dn: StrPath):
        super().__init__()
        _cache_dn = Path(cache_dn)
        self.fn_dic = {
            "meta": _cache_dn / "meta.json",
            "mol_to_frag": _cache_dn / "mol_to_frag.json",
            "frag_to_word": _cache_dn / "frag_to_word.json",
            "frag_vocab": _cache_dn / "frag_vocab.json",
            "word_vocab": _cache_dn / "word_vocab.json"
        }

    def _preprocess_meta(self, chembl_chemreps, is_dev=False):
        toc = Toc()
        logging.info(f"Loading {chembl_chemreps}...")
        with open(chembl_chemreps, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if is_dev:
                reader = islice(reader, 10**5)
            chembl_raw_lst = list(reader)
        logging.info(f"Done! len: {len(chembl_raw_lst)}, toc: {toc():.3f}s")
        meta_lst = []
        for x in chembl_raw_lst:
            # Handling molecules like "c1ccc(cc1)C=O.[Br-].[Br-]"
            smi = max(x["canonical_smiles"].split('.'), key=len)
            meta_lst.append(edict(dict(id=x["chembl_id"], smi=smi)))
        meta_lst = unique_by_key(meta_lst, "smi")
        return meta_lst

    def preprocess(self, chembl_chemreps, n_jobs=4, is_dev=False, 
                    frag_min_freq=1, word_min_freq=1):
        toc = Toc()

        # Preprocess meta
        name = "meta"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            meta_lst = self._preprocess_meta(chembl_chemreps, is_dev)
            logging.info(f"Done! toc: {toc():.3f}s")
            auto_dump(meta_lst, self.fn_dic[name], json_indent=True)
        else:
            logging.info(f"Skip {name}!")
            skip = True
            meta_lst = self.source(name)

        logging.info(f"{'Loaded' if skip else 'Saved'}! "
                     f"{name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
        logging.info(f"{summary_arr(meta_lst, key=lambda x: len(x['smi']))}")

        # Preprocess common flow
        self._preprocess_common_flow(meta_lst, n_jobs, frag_min_freq, word_min_freq)
