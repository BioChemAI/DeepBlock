import argparse
import logging
from numbers import Number
from pathlib import Path
from easydict import EasyDict as edict
from typing import Tuple, Dict, Callable, Union
import numpy as np
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, summary_arr, use_path

from deepblock.evaluation import QEDSA

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_qedsa"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: QED, SA, Lipinski, LogP - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--test-smi-suffix", type=str, default='')
    opt = mix_config(parser, None)
    return opt

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
    ref_smi_fn = saved_dn / f"test_smi{opt.test_smi_suffix}.json"

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/qedsa{opt.suffix}.log")]
    qedsa_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/qedsa{opt.suffix}.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')


    # QED, SA, Lipinski, LogP
    qedsa_dic: Dict[str, Dict[str, float]] = dict()
    toc = Toc()

    sample_smi_dic = auto_load(sample_smi_fn)
    ref_smi_dic = auto_load(ref_smi_fn)
    smi_set = set(smi for smi_lst in sample_smi_dic.values() for smi in smi_lst) | \
        set(ref_smi_dic.values())

    pbar = tqdm(list(smi_set), desc="Compute QED, SA, Lipinski, LogP")
    for smi in pbar:
        try:
            qedsa = QEDSA(smi)
        except Exception as err:
            logging.error(f'{smi} -> {repr(err)}')
        else:
            result = {
                "qed": qedsa.qed(),
                "sa": qedsa.sa(),
                "logp": qedsa.logp()
            }
            lipinski, rules = qedsa.lipinski()
            result["lipinski"] = lipinski
            for i in range(5):
                result[f"lipinski_{i+1}"] = int(rules[i])

            qedsa_dic[smi] = result
        pbar.set_postfix(dict(valid=len(qedsa_dic)))
  
    # Summary
    qedsa_lst = list(qedsa_dic.values())
    for k in qedsa_lst[0].keys():
        s = summary_arr(qedsa_lst, key=lambda x: x[k])
        logging.info(f"{k} -> {s}")

    # Save
    auto_dump(qedsa_dic, qedsa_fn, json_indent=True)
    logging.info(f"Saved -> qedsa_fn: {qedsa_fn}, toc: {toc():.3f}")

    logging.info("Done!")
