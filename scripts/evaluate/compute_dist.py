import argparse
import logging
from numbers import Number
from pathlib import Path
from easydict import EasyDict as edict
from typing import Iterable, Tuple, Dict, Callable, Union
from rdkit import Chem
import numpy as np
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from deepblock.exceptions import RDKitException

from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, rdkit_log_handle, summary_arr, use_path, ignore_exception

from deepblock.evaluation import Dist

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_dist"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Distribution related indicators - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--test-smi-suffix", type=str, default='')
    opt = mix_config(parser, None)
    return opt

def check_reduce(lst_source: Iterable, lst_result: Iterable, name: str):
    a, b = len(lst_source), len(lst_result)
    if a != b:
        logging.warning(f"{name} -> {a}-{b}={a-b} failed!")

if __name__ == '__main__':
    opt = parse_opt()

    # Define Path
    opt.train_id = opt.base_train_id
    if opt.baseline:
        saved_dn = Path(f"saved/baseline/{opt.train_id}")
    else:
        saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    sample_smi_fn = saved_dn / f"sample/smi{opt.suffix}.json"
    train_smi_fn = saved_dn / f"train_smi.json"
    ref_smi_fn = saved_dn / f"test_smi{opt.test_smi_suffix}.json"

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/dist{opt.suffix}.log")]
    dist_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/dist{opt.suffix}.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Dist
    dist_dic: Dict[str, Dict[str, Dict]] = dict()
    toc = Toc()

    sample_smi_dic = auto_load(sample_smi_fn)
    train_smi_dic = auto_load(train_smi_fn)
    ref_smi_dic = auto_load(ref_smi_fn)

    train_smi_lst = list(train_smi_dic.values())

    # Cache all fingerprints
    smi_set = set(smi for smi_lst in sample_smi_dic.values() for smi in smi_lst) | \
        set(train_smi_lst) | set(ref_smi_dic.values())
    smi_lst = list(smi_set)
    
    @ignore_exception
    def _f_fp_sfp(smi):
        with rdkit_log_handle() as rdLog:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise RDKitException(rdLog())
            fp = Dist.fp(mol)
            sfp = Dist.scaffold_fp(mol)
        return fp, sfp
     
    _r_fp_sfp_lst = list(tqdm(map(_f_fp_sfp, smi_lst), total=len(smi_lst), desc="Calculate fp"))
    fp_sfp_lst = [x for x in _r_fp_sfp_lst if x is not None]
    check_reduce(_r_fp_sfp_lst, fp_sfp_lst, "fp_sfp_lst")

    fp_lst, sfp_lst = zip(*fp_sfp_lst)
    smi_to_fp = dict(zip(smi_lst, fp_lst))
    smi_to_sfp = dict(zip(smi_lst, sfp_lst))

    train_fp_lst = [smi_to_fp[smi] for smi in train_smi_lst if smi in smi_to_fp]
    check_reduce(train_smi_lst, train_fp_lst, "train_fp_lst")
    train_sfp_lst = [smi_to_sfp[smi] for smi in train_smi_lst if smi in smi_to_fp]
    check_reduce(train_smi_lst, train_sfp_lst, "train_sfp_lst")

    pbar = tqdm(sample_smi_dic.keys(), desc="Compute dist")

    for meta_id in pbar:
        prd_smi_lst = sample_smi_dic[meta_id]
        prd_fp_lst = [smi_to_fp[smi] for smi in prd_smi_lst if smi in smi_to_fp]
        check_reduce(prd_smi_lst, prd_fp_lst, "prd_fp_lst")
        prd_sfp_lst = [smi_to_sfp[smi] for smi in prd_smi_lst if smi in smi_to_sfp]
        check_reduce(prd_smi_lst, prd_sfp_lst, "prd_sfp_lst")
        
        ref_smi = ref_smi_dic[meta_id]
        ref_fp = smi_to_fp[ref_smi]
        ref_sfp = smi_to_fp[ref_smi]
        assert (ref_fp is not None) and (ref_sfp is not None), f"Cannot continue -> ref_smi: {ref_smi}"

        _dic = dict(
            similarity = Dist.similarity(fps=prd_fp_lst),
            ref_similarity = Dist.ref_similarity(ref_fp=ref_fp, prd_fps=prd_fp_lst),
            novelty_fp = Dist.novelty_fp(train_fps=train_fp_lst, prd_fps=prd_fp_lst),
            novelty_smi = Dist.novelty_smi(train_smi_lst=train_smi_lst, prd_smi_lst=prd_smi_lst),
            scaffold_similarity = Dist.similarity(fps=prd_sfp_lst),
            scaffold_ref_similarity = Dist.ref_similarity(ref_fp=ref_sfp, prd_fps=prd_sfp_lst),
            scaffold_novelty_fp = Dist.novelty_fp(train_fps=train_sfp_lst, prd_fps=prd_sfp_lst)
        )
        
        pbar.set_postfix(_dic)
        dist_dic[meta_id] = _dic

    # Save
    auto_dump(dist_dic, dist_fn, json_indent=True)
    logging.info(f"Saved -> dist_fn: {dist_fn}, toc: {toc():.3f}")

    logging.info("Done!")
