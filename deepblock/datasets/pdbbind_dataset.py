from collections import Counter
from itertools import islice, chain
from pathlib import Path
from typing import Dict, List
import re
import logging
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from .raw_third_dataset import RawThirdDataset
from ..utils import StrPath, Toc, get_file_size, \
    pretty_kv, auto_dump, summary_arr, complex_to_aa

class PDBbindDataset(RawThirdDataset):
    r"""PDBbind dataset with
    
    - Protein: Amino acid sequence, center of mass for each amino acid.
    - Ligand: Fragment, word.

    Source:

    Structure for `meta`

    ```json
    {
        "id": "1add",
        "refined": true,
        "smi": "Nc1ccnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
        "protein": "PDBbind_v2020_refined/refined-set/1add/1add_protein.pdb",
        "ligand": "PDBbind_v2020_refined/refined-set/1add/1add_ligand.sdf",
        "pocket": "PDBbind_v2020_refined/refined-set/1add/1add_pocket.pdb",
        "resolution": 2.4,
        "release_year": 1994,
        "binding_data": "Ki=0.18uM",
        "reference": "1add.pdf",
        "ligand_name": "(1DA)",
        "uniprot_id": "P03958",
        "protein_name": "ADENOSINE DEAMINASE"
    }
    ```

    Structure for `aa`

    ```json
    {
        "184l": {
            "id": "A", 
            "len": 162,                         // len(seq)
            "seq": "MNIFEMLR...",               // length = L
            "ids": [1, 2, 3, "<...>"],          // length = L
            "pos": "<np.ndarray>",              // shape = [L, 3]
            "pocket_cent": "<np.ndarray>",      // shape = [3]
            "pocket_dist": "<np.ndarray>"       // shape = [L]
        },
        '<id>': {},
        ...
    }
    ```
    """

    KNOWN_TOTAL = 19443

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
            "word_vocab": _cache_dn / "word_vocab.json",
            "pick_set": _cache_dn / "pick_set.json"
        }

    def _preprocess_meta(self, pdbbind_dir: Path=None, 
                        is_dev=False) -> List[Dict]:

        reg_1 = re.compile(
            r'^(?P<id>\w{4})  (?P<resolution>.{4})  (?P<release_year>\d{4})  (?P<binding_data>\S+)\s+\/\/ (?P<reference>\S+) (?P<ligand_name>.+)$',
            flags=re.MULTILINE)
        reg_2 = re.compile(
            r'^(?P<id>\w{4})  (?P<release_year>\d{4})  (?P<uniprot_id>\S{6})  (?P<protein_name>.+)$',
            flags=re.MULTILINE)
        reg_3 = re.compile(r'[+-]?((\d+\.?\d*)|(\.\d+))')
        
        meta_dic = {}
        fn = pdbbind_dir / "PDBbind_v2020_plain_text_index/index/INDEX_general_PL.2020"
        with open(fn, "r") as f:
            string = f.read()
        
        matches = reg_1.finditer(string)
        for m in matches:
            meta_dic[m["id"]] = {
                "resolution": float(m["resolution"]) if reg_3.match(m["resolution"]) else m["resolution"].lstrip(),
                "release_year": int(m["release_year"]),
                "binding_data": m["binding_data"],
                "reference": m["reference"],
                "ligand_name": m["ligand_name"],
            }
        if len(meta_dic) != self.KNOWN_TOTAL:
            logging.warning(f"The total amount {len(meta_dic)} is not the same as known {self.KNOWN_TOTAL}!")

        fn = pdbbind_dir / "PDBbind_v2020_plain_text_index/index/INDEX_general_PL_name.2020"
        with open(fn, "r") as f:
            string = f.read()
        matches = reg_2.finditer(string)
        for m in matches:
            meta_dic[m["id"]] = {
                **meta_dic[m["id"]],
                "uniprot_id": m["uniprot_id"],
                "protein_name": m["protein_name"],
            }

        refined_id_set = set()
        fn = pdbbind_dir / "PDBbind_v2020_plain_text_index/index/INDEX_refined_set.2020"
        with open(fn, "r") as f:
            string = f.read()
        matches = reg_1.finditer(string)
        for m in matches:
            refined_id_set.add(m["id"])
            
        if is_dev:
            meta_dic = {k: meta_dic[k] for k in chain(
                islice((x for x in meta_dic if x not in refined_id_set), 100), 
                islice((x for x in meta_dic if x in refined_id_set), 100)
            )}

        meta_lst = []
        pbar = tqdm(meta_dic.items(), desc=f"Preprocess [meta]")
        status = edict(dict(found=0, lost=0, bad=0))
        for cid, val in pbar:
            try:
                dir_prefix = "PDBbind_v2020_refined/refined-set" \
                    if cid in refined_id_set else "PDBbind_v2020_other_PL/v2020-other-PL"
                dic = edict(
                    id = cid, refined = cid in refined_id_set, smi = None,
                    protein = f"{dir_prefix}/{cid}/{cid}_protein.pdb",
                    ligand = f"{dir_prefix}/{cid}/{cid}_ligand.sdf", 
                    pocket = f"{dir_prefix}/{cid}/{cid}_pocket.pdb", 
                    **val
                )
                for k in ("protein", "ligand", "pocket"):
                    fn = pdbbind_dir / dic[k]
                    if not fn.exists():
                        raise self._LostException(f"Lost {k}: {fn}")

                try:
                    dic.smi = complex_to_aa.sdf_to_smi(pdbbind_dir / dic.ligand)
                    assert dic.smi, "Empty SMILES from SDF"
                except Exception as error:
                    raise Exception(f"Bad ligand: {dic.ligand}\n{error}")

            except Exception as error:
                # logging.error(error)
                if isinstance(error, self._LostException):
                    status.lost += 1
                else:
                    status.bad += 1
            else:
                meta_lst.append(dic)
                status.found += 1

            pbar.set_postfix(status)

        logging.info(f"status -> {pretty_kv(status)}")

        meta_lst = sorted(meta_lst, key=lambda x: x.id)
        return meta_lst

    def preprocess(self, pdbbind_dir: Path,
                    n_jobs=4, is_dev=False, 
                    frag_min_freq=1, word_min_freq=1):
        toc = Toc()

        # Preprocess meta
        name = "meta"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            meta_lst = self._preprocess_meta(pdbbind_dir, is_dev)
            logging.info(f"Done! toc: {toc():.3f}s")
            auto_dump(meta_lst, self.fn_dic[name], json_indent=True)
        else:
            logging.info(f"Skip {name}!")
            skip = True
            meta_lst = self.source(name)

        logging.info(f"{'Loaded' if skip else 'Saved'}! "
                     f"{name} -> {self.fn_dic[name]}, toc: {toc():.3f}s")
        logging.info(f'refined -> {sum(x.refined for x in meta_lst)}/{len(meta_lst)}')
        logging.info(f"{summary_arr(meta_lst, key=lambda x: len(x['smi']))}")

        # Preprocess complex_to_aa
        name = "complex_to_aa"
        skip = False
        toc()
        if not self.fn_dic[name].exists():
            logging.info(f"Start {name}!")
            complex_to_aa_dic = self._preprocess_complex_to_aa(pdbbind_dir, meta_lst, 
                                                               n_jobs=n_jobs, ignore_warning=not is_dev)
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
