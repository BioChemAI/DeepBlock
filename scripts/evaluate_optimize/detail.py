"""
Create a large table with one sample per row
"""

import argparse
import logging
from numbers import Number
from pathlib import Path
import warnings
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
import hashlib

from deepblock.utils import Toc, auto_dump, auto_load, auto_loadm, ifn, init_logging, \
    mix_config, use_path, sequential_file_hash

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_summary.optimize"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Detail for optimize - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--docking-suffix", type=str)
    opt = mix_config(parser, None)
    return opt

class DetailCycle:

    def __init__(self, **fns) -> None:
        self.fns = edict(fns)

    def compute(self):
        prd_res_dic = auto_load(self.fns.prd_res_fn)
        docking_score_dic = auto_load(self.fns.docking_score_fn)             # [{id(hash), score}, ...]      Data will be missing!!
        docking_lookup_dic = edict(auto_load(self.fns.docking_lookup_fn))    # {ref: {id: hash, ...}, prd: {id: {smi: hash, ...}, ...}}
        qedsa_dic = auto_load(self.fns.qedsa_fn)                             # {smi: {qed, sa, lipinski, logp, logp_1, ...}, ...}    Data will be missing!!
        dist_dic = auto_load(self.fns.dist_fn)

        _fn: Path = self.fns.docking_input_fn
        if _fn.suffix == '.auto':
            found_flag = False
            for docking_input_suffix in ('.tar.gz', '.7z'):
                _fn2 = _fn.with_suffix(docking_input_suffix)
                if _fn2.exists():
                    _fn = _fn2
                    found_flag = True
                    break
            assert found_flag

        docking_job_id = sequential_file_hash(_fn, hashlib.sha256()).hexdigest()
        hash_to_score = {x['id']: x['affinity'] for x in docking_score_dic if x['affinity'] < 0}

        for gcid, dic in prd_res_dic.items():
            gid, cid = gcid.split(';')
            for mod in ("ref", "prd"):
                smi = dic[f"{mod}_smi"]
                d = edict(gid=gid, cid=cid, smi=smi, ref=1 if mod == 'ref' else 0)
                d.docking_job_id = docking_job_id
                d.docking_cid = docking_lookup_dic[gcid][mod]
                d.affinity = hash_to_score.get(d.docking_cid, None)
                qedsa_item = qedsa_dic[d.smi]
                for name in ('qed', 'sa', 'lipinski', 'logp', *(f'lipinski_{i+1}' for i in range(5))):
                    d[name] = qedsa_item[name] if qedsa_item is not None else None
                for name in ("similarity", "scaffold_similarity"):
                    d[name] = dist_dic[gcid][name]
                yield d

if __name__ == '__main__':
    opt = parse_opt()
    opt.docking_suffix = ifn(opt.docking_suffix, opt.suffix)

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    detail_cycle = DetailCycle(
        prd_res_fn = saved_dn / f"optimize/prd_res{opt.suffix}.json",
        docking_input_fn = saved_dn / f"optimize/batch_docking_input{opt.docking_suffix}.auto",
        docking_score_fn = saved_dn / f"optimize/batch_docking_score{opt.docking_suffix}.json",
        docking_lookup_fn = saved_dn / f"optimize/batch_docking_lookup{opt.docking_suffix}.json",
        qedsa_fn = saved_dn / f"optimize/qedsa{opt.suffix}.json",
        dist_fn = saved_dn / f"optimize/dist{opt.suffix}.json",
    )

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"optimize/detail{opt.suffix}.log")]
    final_json_fn = use_path(file_path=saved_dn / f"optimize/detail{opt.suffix}.json")
    final_csv_fn = final_json_fn.with_suffix('.csv')

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Compute
    toc = Toc()
    detail_records = list(detail_cycle.compute())

    # Save
    auto_dump(detail_records, final_json_fn, json_indent=True)
    logging.info(f"Saved -> final_json_fn: {final_json_fn}, toc: {toc():.3f}")

    df = pd.DataFrame(detail_records)
    df.to_csv(final_csv_fn, index=False)
    logging.info(f"Saved -> final_csv_fn: {final_csv_fn}, toc: {toc():.3f}")

    logging.info("Done!")
