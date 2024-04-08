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
STAGE = "evaluate_compute_dist.optimize"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Distribution related indicators - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--smi-train-id", type=str)
    # --smi-train-id dry_run_crossdocked_smi
    # --smi-train-id dry_run_chembl_smi
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
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    optimize_result_fn = saved_dn / f"optimize/prd_res{opt.suffix}.json"

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"optimize/dist{opt.suffix}.log")]
    dist_fn = use_path(file_path=saved_dn / f"optimize/dist{opt.suffix}.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Dist
    dist_dic: Dict[str, Dict[str, Dict]] = dict()
    toc = Toc()

    optimize_result_dic = auto_load(optimize_result_fn)

    # Cache all fingerprints
    smi_set = set(smi for x in optimize_result_dic.values() for smi in (x["ref_smi"], x["prd_smi"]))
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

    pbar = tqdm(optimize_result_dic.items(), desc="Compute dist")

    for k, v in pbar:

        ref_fp = smi_to_fp[v["ref_smi"]]
        ref_sfp = smi_to_sfp[v["ref_smi"]]
        prd_fp = smi_to_fp[v["prd_smi"]]
        prd_sfp = smi_to_sfp[v["prd_smi"]]

        assert all(x is not None for x in (ref_fp, ref_sfp, prd_fp, prd_sfp)), f"Cannot continue -> {k}"

        _dic = dict(
            similarity = Dist._similarity(ref_fp, prd_fp),
            scaffold_similarity = Dist._similarity(ref_sfp, prd_sfp),
        )
        
        dist_dic[k] = _dic

    # Save
    auto_dump(dist_dic, dist_fn, json_indent=True)
    logging.info(f"Saved -> dist_fn: {dist_fn}, toc: {toc():.3f}")

    logging.info("Done!")
