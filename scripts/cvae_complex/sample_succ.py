"""Sample CVAE Complex Model
"""

import argparse
from itertools import islice
import logging
from pathlib import Path
import pandas as pd
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import name_to_dataset_cls, ComplexAADataset
from deepblock.api import APICVAEComplex, ChemAssertTypeEnum, ChemAssertActionEnum, ChemAssertException
from deepblock.utils import auto_dump, init_logging, \
    mix_config, init_random, pretty_kv, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "sample_succ"


def parse_opt():
    parser = argparse.ArgumentParser(description='Sample: CVAE Complex')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--base-weight-choice", type=str, default="latest")
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230609)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--max-attempts", type=int, default=1<<7)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    if opt.suffix is None:
        opt.suffix = f'_{opt.max_attempts}'
    else:
        opt.suffix = ''

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    sample_log_fn = use_path(file_path=saved_dn / f"{STAGE}/{STAGE}{opt.suffix}.log")
    sample_smi_fn = use_path(file_path=saved_dn / f"{STAGE}/smi{opt.suffix}.json")

    # Initialize log, device
    init_logging((log_fn, sample_log_fn))
    logging.info(f'opt: {opt}')

    # Learner
    api = APICVAEComplex(
        saved_dn=saved_dn, 
        weight_choice=opt.base_weight_choice,
        device=opt.device
    )
    learner = api.learner

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    split_pro_dic=learner.upstream_opt.split_pro
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    _dataset_opt = dict(d=_dataset,
                        rel_fn=learner.rel_fn,
                        x_vocab=learner.x_vocab,
                        x_max_len=learner.x_max_len,
                        c_max_len=learner.c_max_len,
                        split_pro_dic=split_pro_dic,
                        is_dev=False)
    test_set = ComplexAADataset(**_dataset_opt, split_key='test')
    logging.info(f"len(test_set): {len(test_set)}")

    item_smi_dic = dict()

    pbar = tqdm(total=len(test_set)*opt.max_attempts, desc="Sample")
    for epoch, item in enumerate(test_set):
        desc = 'Sample'
        # print(f'epoch -> {epoch}/{len(test_set) - 1}, {item.id}')
        sampler = api.chem_sample(item, batch_size=opt.batch_size,
                                  assert_types={ },
                                  assert_actions={ ChemAssertActionEnum.RETURN },
                                  max_attempts=opt.max_attempts,
                                  max_attempts_exceeded_action='raise',
                                  desc=item.id)
        
        res_lst = []
        for res in islice(sampler, opt.max_attempts):
            pbar.update(1)
            res_lst.append(res.smi)

        succ = len(set(smi for smi in res_lst if smi is not None))/len(res_lst)*100
        pbar.set_postfix(dict(epoch=epoch, item_id=item.id, succ=succ))
        item_smi_dic[item.id] = res_lst

    pbar.close()

    auto_dump(item_smi_dic, sample_smi_fn, json_indent=True)
    logging.info(f"Saved item_smi_dic -> {sample_smi_fn}")
    logging.info("Done!")
