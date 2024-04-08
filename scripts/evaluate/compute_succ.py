import argparse
from itertools import chain
import logging
from numbers import Number
from pathlib import Path
from typing import Dict
import pandas as pd
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, use_path
from deepblock.api import embed_smi_cached

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_succ"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Succ of sample - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    sample_smi_fn = saved_dn / f"sample_succ/smi{opt.suffix}.json"
    train_smi_fn = saved_dn / f"train_smi.json"

    bpt = auto_load(saved_dn / "opt.json")
    opt.pre_train_id = bpt["base_train_id"]

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/succ{opt.suffix}.log")]
    succ_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/succ{opt.suffix}.json")
    succ_mean_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/succ_mean{opt.suffix}.csv")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    if opt.pre_train_id is not None:
        logging.info("Detect pretrain SMILES!")
        opt.pretrain_id = opt.pre_train_id
        pretrain_smi_fn = Path(f"saved/{MODEL_TYPE}/{opt.pretrain_id}") / f"train_smi.json"
    else:
        pretrain_smi_fn = None

    # Succ
    toc = Toc()
    succ_dic: Dict[str, Dict[str, Number]] = dict()
    
    train_smi_dic = auto_load(train_smi_fn)
    train_smi_lst = list(train_smi_dic.values())
    if pretrain_smi_fn is not None:
        pretrain_smi_dic = auto_load(pretrain_smi_fn)
        train_smi_lst.extend(pretrain_smi_dic.values())
    train_smi_set = set(filter(lambda smi: smi is not None, train_smi_lst))

    sample_smi_dic = auto_load(sample_smi_fn)

    def val_nov_uni(smi_lst):
        d = dict()
        val_lst = list(filter(lambda smi: smi is not None, smi_lst))
        uni_set = set(val_lst)
        nov_lst = [smi for smi in val_lst if smi not in train_smi_set]
        emb_lst = [smi for smi in val_lst if embed_smi_cached(smi)]
        d["validity"] = len(val_lst) / len(smi_lst)
        d["novelty"] = len(nov_lst) / len(val_lst)
        d["embeddable"] = len(emb_lst) / len(val_lst)
        d["uniqueness"] = len(uni_set) / len(val_lst)
        
        return d

    for item_id, smi_lst in tqdm(sample_smi_dic.items()):
        succ_dic[item_id] = val_nov_uni(smi_lst)

    # Save
    auto_dump(succ_dic, succ_fn, json_indent=True)
    logging.info(f"Saved -> succ_fn: {succ_fn}, toc: {toc():.3f}")

    smi_lst = list(chain.from_iterable(sample_smi_dic.values()))
    d_all = val_nov_uni(smi_lst)

    df = pd.DataFrame.from_dict(succ_dic, orient='index')
    succ_df = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'std_biased': df.std(ddof=0),
        'all': d_all
    })

    logging.info(f"succ_df:\n{succ_df}")
    logging.info(f"succ_df*100:\n{(succ_df*100).round(2)}")
    succ_df.to_csv(succ_mean_fn)

    logging.info("Done!")
