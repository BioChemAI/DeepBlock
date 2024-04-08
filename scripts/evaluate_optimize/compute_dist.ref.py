import argparse
from itertools import combinations
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
STAGE = "evaluate_compute_dist.optimize.ref"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Distribution related indicators - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--suffix", type=str, default='')
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
              use_path(file_path=saved_dn / f"optimize/dist.ref{opt.suffix}.log")]
    dist_fn = use_path(file_path=saved_dn / f"optimize/dist.ref{opt.suffix}.msgpack")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Dist
    dist_dic: Dict[str, Dict[str, Dict]] = dict()
    toc = Toc()

    optimize_result_dic = auto_load(optimize_result_fn)

    # Cache all fingerprints
    smi_set = set(x["ref_smi"] for x in optimize_result_dic.values())
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

    ref_smi_lst = [v["ref_smi"] for v in optimize_result_dic.values()]
    comb_smi_lst = list(combinations(ref_smi_lst, 2))

    similarity_lst = []
    for smi_x, smi_y in tqdm(comb_smi_lst, desc="Compute dist"):
        similarity_lst.append((
            Dist._similarity(smi_to_fp[smi_x], smi_to_fp[smi_y]),
            Dist._similarity(smi_to_sfp[smi_x], smi_to_sfp[smi_y]),
        ))

    similarity_arr = np.array(similarity_lst)

    # Save
    auto_dump(similarity_arr, dist_fn, json_indent=True)
    logging.info(f"Saved -> dist_fn: {dist_fn}, toc: {toc():.3f}")

    logging.info("Done!")
