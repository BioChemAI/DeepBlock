"""Molecular optimization by CVAE Complex Model

Randomly select molecules in the ChEMBL and optimize them as target molecules
"""

import argparse
from collections import OrderedDict
import hashlib
import logging
from pathlib import Path
import random
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import get_complex_aa_dataset_by_learner, \
    name_to_dataset_cls, ComplexAADataset, ComplexAAItem
from deepblock.utils import auto_dump, auto_load, init_logging, \
    mix_config, init_random, pretty_kv, use_path
from deepblock.api import APICVAEComplex, embed_smi_cached, detokenize_cached

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "optimize"


def parse_opt():
    parser = argparse.ArgumentParser(description='Optimize: CVAE Complex')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--base-weight-choice", type=str, default="latest")
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--chembl-cached-dn", type=str,
                        default="saved/preprocess/chembl")
    parser.add_argument("--c-include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--x-include", type=str,
                        choices=name_to_dataset_cls.keys(), default='chembl')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--complex-id", type=str)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    if opt.suffix is None:
        opt.suffix = f'_{opt.num_samples}'
        if opt.complex_id is not None:
            opt.suffix += f'_only-{hashlib.sha256(opt.complex_id.encode()).hexdigest()[:4]}'
    else:
        opt.suffix = ''

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    sample_log_fn = use_path(file_path=saved_dn / f"{STAGE}/{STAGE}{opt.suffix}.log")

    ref_res_fn = use_path(file_path=saved_dn / f"{STAGE}/ref_res{opt.suffix}.json")
    prd_res_fn = use_path(file_path=saved_dn / f"{STAGE}/prd_res{opt.suffix}.json")
    human_readable_fn = use_path(file_path=saved_dn / f"{STAGE}/human_readable{opt.suffix}.json")

    # Initialize log, device, config, wandb
    init_logging((log_fn, sample_log_fn))
    logging.info(f'opt: {opt}')

    # Reproducibility
    init_random(opt.random_seed)

    # Learner
    api = APICVAEComplex(
        saved_dn=saved_dn, 
        weight_choice=opt.base_weight_choice,
        device=opt.device
    )
    learner = api.learner

    # Dataset
    x_set = get_complex_aa_dataset_by_learner(
        opt.x_include, opt[f"{opt.x_include}_cached_dn"], 'valid', learner)
    logging.info(f"len(x_set): {len(x_set)}")
    c_set = get_complex_aa_dataset_by_learner(
        opt.c_include, opt[f"{opt.c_include}_cached_dn"], 'test', learner)
    logging.info(f"len(c_set): {len(c_set)}")

    if opt.complex_id is not None:
        c_set.raw_lst = [next(x for x in c_set.raw_lst if x['id'] == opt.complex_id)]
        logging.info(f"Only pick {opt.complex_id}, len(c_set): {len(c_set)}")

    x_id_smi_dic = x_set.id_smi_dic
    c_id_smi_dic = c_set.id_smi_dic

    # Building a task queue
    if not ref_res_fn.exists():
        desc = f"[{STAGE}] Pick"
        x_random = random.Random(20230325)
        x_item_lst = []
        ref_res_dic = dict()
        assert len(x_set) >= opt.num_samples, "Are you kidding? Not enough!"
        x_remain_inds = set(range(len(x_set)))
        pbar = tqdm(total=opt.num_samples, desc=desc)
        while len(x_item_lst) < opt.num_samples:
            res = edict(seq=None, smi=None, frags=None)
            ind = x_random.choice(list(x_remain_inds))
            x_remain_inds.remove(ind)
            x_item = x_set[ind]
            x_smi = x_id_smi_dic[x_item.id]
            try:
                assert len(x_item.x) < x_set.x_max_len, "Too long"
                res.seq = x_item.x[1:-1]
                res.frags = x_set.x_vocab.itos(res.seq)
                res.smi = detokenize_cached(res.frags)
                
                # assert x_smi == res.smi, "Inconsistent with original SMILES"
                assert embed_smi_cached(res.smi), "Embed failed"
            except Exception as err:
                logging.warning(f"ChEMBL: {x_item.id}, {x_smi} -> {repr(err)}")
            else:
                ref_res_dic[x_item.id] = res
                x_item_lst.append(x_item)
                pbar.update()
        pbar.close()
        auto_dump(ref_res_dic, ref_res_fn)
        logging.info(f"Saved! {ref_res_fn}")
    else:
        ref_res_dic = auto_load(ref_res_fn)
        x_item_dic = {x_item.id: x_item for x_item in x_set if x_item.id in ref_res_dic.keys()}
        x_item_lst = [x_item_dic[x_item_id] for x_item_id in ref_res_dic.keys()]
        assert len(x_item_lst) == opt.num_samples
        logging.info(f"Loaded! {ref_res_fn}")

    desc = f"[{STAGE}] Sample"
    task_gen = (
        ComplexAAItem(
            id=f"{c_item.id};{x_item.id}",
            x=x_item.x,
            c=c_item.c,
            # c_rel=c_item.c_rel # It doesn't matter
        ) for c_item in c_set for x_item in x_item_lst)
    
    # Start Optimize
    pbar = tqdm(total=len(x_item_lst)*len(c_set), desc=desc)
    optimizer = api.chem_optimize(task_gen, opt.batch_size)
    prd_res_dic = dict()
    for cid, res in optimizer:
        dic = dict()
        dic.update({f"ref_{k}": v for k, v in ref_res_dic[cid.split(';')[1]].items()})
        dic.update({f"prd_{k}": v for k, v in res.items()})
        prd_res_dic[cid] = dic
        succ = len(prd_res_dic) / optimizer.num_attempts*100 if optimizer.num_attempts > 0 else 0
        pbar.update()
        pbar.set_postfix({"succ": f'{succ:.3f}%'})
    pbar.close()

    def f_format_num_lst(lst): 
        return ','.join([f'{num:>{5}}' for num in lst])
    def f_format_str_lst(lst): 
        return ', '.join(lst)

    auto_dump(prd_res_dic, prd_res_fn)
    logging.info(f"Saved! {prd_res_fn}")

    k_lst = sorted(prd_res_dic.keys())
    human_readable_dic = OrderedDict()
    for k in k_lst:
        v = edict(prd_res_dic[k])
        human_readable_dic[k] = dict(
            ref_seq=f_format_num_lst(v.ref_seq),
            prd_seq=f_format_num_lst(v.prd_seq),
            ref_smi=v.ref_smi,
            prd_smi=v.prd_smi,
            ref_frags=f_format_str_lst(v.ref_frags),
            prd_frags=f_format_str_lst(v.prd_frags)
        )

    auto_dump(human_readable_dic, human_readable_fn, json_indent=True)
    logging.info(f"Saved! {human_readable_fn}")

    logging.info("Done!")
