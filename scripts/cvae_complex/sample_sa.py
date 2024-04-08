"""Sample with SA for CVAE Complex Model
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from functools import lru_cache, partial
import numpy as np
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import get_complex_aa_item_by_learner, name_to_dataset_cls, \
    ComplexAADataset, ComplexAAItem
from deepblock.utils import auto_dump, auto_load, init_logging, \
    mix_config, init_random, pretty_kv, use_path, use_memory
from deepblock.api import APICVAEComplex4SA, APIRegressTox
from deepblock.evaluation import QEDSA

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "sample_sa"

memory = use_memory()

def parse_opt():
    parser = argparse.ArgumentParser(description='Sample with SA: CVAE Complex')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--base-weight-choice", type=str, default="latest")
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--chembl-cached-dn", type=str,
                        default="saved/preprocess/chembl")
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--complex-id", type=str, required=True)
    opt = mix_config(parser, None)
    return opt

@memory.cache
def get_dataset_item(name, cached_dn, split_key, cid, *args, **kwargs) -> ComplexAAItem:
    item = get_complex_aa_item_by_learner(name, cached_dn, split_key, learner, cid)
    return item


if __name__ == '__main__':
    opt = parse_opt()

    if opt.suffix is None:
        opt.suffix = f'_{opt.num_steps}-{opt.num_samples}'
        if opt.complex_id is not None:
            opt.suffix += f'_only-{hashlib.sha256(opt.complex_id.encode()).hexdigest()[:4]}'
    else:
        opt.suffix = ''

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    sample_log_fn = use_path(file_path=saved_dn / f"{STAGE}/{STAGE}{opt.suffix}.log")
    res_fn = use_path(file_path=saved_dn / f"{STAGE}/res{opt.suffix}.json")
    info_fn = use_path(file_path=saved_dn / f"{STAGE}/info{opt.suffix}.json")

    # Initialize log, device, config, wandb
    init_logging((log_fn, sample_log_fn))
    logging.info(f'opt: {opt}')

    # API
    api = APICVAEComplex4SA(
        saved_dn=saved_dn, 
        weight_choice=opt.base_weight_choice,
        device=opt.device
    )
    learner = api.learner

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    item = get_dataset_item(
        opt.include, opt[f"{opt.include}_cached_dn"], 'test', opt.complex_id,
        opt.base_train_id, opt.base_weight_choice)
    logging.info(f"Get -> item: {item}")

    # Fitness
    _TSMILES = APICVAEComplex4SA.TSMILES
    _TPopuSA = APICVAEComplex4SA.TPopuSA
    _TFitness = APICVAEComplex4SA.TFitness

    @lru_cache(maxsize=1<<16)
    def qed_fn_cached(smi: _TSMILES):
        return QEDSA(smi).qed()
    tox_fn_cached = APIRegressTox().chem_predict_cached

    def fitness_fn(popu_lst: _TPopuSA) ->_TFitness:
        fitness_dic = dict()
        for smi in set(popu_lst):
            tox = tox_fn_cached(smi)
            qed = qed_fn_cached(smi)
            fitness_dic[smi] = {
                "tox": tox,
                "qed": qed,
                "scaled": np.average([
                    np.clip(0.5*tox-0.5, 0, 1),
                    np.clip(-qed+1, 0, 1)
                ], weights=[
                    3, 2
                ])
            }
        return fitness_dic

    # SA
    api.init_popu(
        c_item=item, fitness_fn=fitness_fn,
        popu_size=opt.num_samples, T0=1, D=0.8,
        batch_size=opt.batch_size,
    )

    # Run
    logging.info(f"Start")
    res_dic = api.run(opt.num_steps)
    logging.info(f"res_dic: {json.dumps(res_dic[:10])}")

    # Save
    auto_dump(res_dic, res_fn, json_indent=True)
    logging.info(f"Saved! res_dic: {res_fn}")

    info_dic = api.info()
    info_dic['opt'] = opt
    auto_dump(info_dic, info_fn, json_indent=True)
    logging.info(f"Saved! info_dic: {info_fn}")

    logging.info("Done!")
