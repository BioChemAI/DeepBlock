"""Sample CVAE Complex Model
"""

import argparse
from itertools import islice
import logging
from pathlib import Path
import warnings
import torch
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import name_to_dataset_cls, ComplexAADataset
from deepblock.api import APICVAEComplex, ChemAssertTypeEnum, ChemAssertActionEnum
from deepblock.utils import auto_dump, init_logging, \
    mix_config, init_random, pretty_kv, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "sample"


def parse_opt():
    parser = argparse.ArgumentParser(description='Sample: CVAE Complex')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--base-weight-choice", type=str, default="latest")
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--pdbbind-cached-dn", type=str,
                        default="saved/preprocess/pdbbind")
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1<<12)
    parser.add_argument("--validate-mol", action="store_true")
    parser.add_argument("--embed-mol", action="store_true")
    parser.add_argument("--unique-mol", action="store_true")
    parser.add_argument("--groundtruth-rel", action="store_true")
    parser.add_argument("--force-mean", action="store_true")
    parser.add_argument("--post-suffix", type=str)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    if opt.unique_mol and (not opt.validate_mol):
        raise Exception("Args: --unique-mol must --validate-mol")
    if opt.embed_mol and (not opt.validate_mol):
        raise Exception("Args: --embed-mol must --validate-mol")
    
    opt_to_assert = {
        "validate_mol": ('v', ChemAssertTypeEnum.VALIDATE),
        "embed_mol": ('e', ChemAssertTypeEnum.EMBED),
        "unique_mol": ('u', ChemAssertTypeEnum.UNIQUE),
    }
    _assert_lst = [v for k, v in opt_to_assert.items() if opt[k]]
    assert_suffixes, assert_types = zip(*_assert_lst)

    assert not (opt.force_mean and opt.groundtruth_rel), \
        "Args: --force-mean and --groundtruth-rel are mutually exclusive"

    if opt.suffix is None:
        opt.suffix = f'_{opt.num_samples}'
        opt.suffix += ''.join(assert_suffixes)
        if opt.groundtruth_rel:
            opt.suffix += '_rel'
        if opt.force_mean:
            opt.suffix += '_mean'
    else:
        opt.suffix = ''

    if opt.post_suffix is None:
        opt.post_suffix = ''

    opt.suffix += opt.post_suffix

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}{opt.post_suffix}.log")
    sample_log_fn = use_path(file_path=saved_dn / f"{STAGE}/{STAGE}{opt.suffix}.log")
    sample_fn_dic = edict(dict(
        status=use_path(file_path=saved_dn / f"{STAGE}/status{opt.suffix}.json"),
        frags=use_path(file_path=saved_dn / f"{STAGE}/frags{opt.suffix}.json"),
        smi=use_path(file_path=saved_dn / f"{STAGE}/smi{opt.suffix}.json"),
        attn=use_path(file_path=saved_dn / f"{STAGE}/attn{opt.suffix}.msgpack"),
    ))
    test_smi_fn = use_path(file_path=saved_dn / f"test_smi{opt.post_suffix}.json")

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
    model = learner.model

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    split_pro_dic=learner.upstream_opt.split_pro
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    _dataset_opt = dict(d=_dataset,
                        rel_fn=learner.rel_fn,
                        # x_vocab=learner.x_vocab,
                        # # If it is a different dataset, target molecules that 
                        # # do not meet the x_vocab criteria will be ignored, 
                        # # which is not necessary for sampling
                        x_max_len=learner.x_max_len,
                        c_max_len=learner.c_max_len,
                        split_pro_dic=split_pro_dic,
                        is_dev=False)
    test_set = ComplexAADataset(**_dataset_opt, split_key='test')
    logging.info(f"len(test_set): {len(test_set)}")
    auto_dump(test_set.id_smi_dic, test_smi_fn)
    logging.info(f"Backup smi to {test_smi_fn}, {test_smi_fn}")

    # Loop
    sample_res_dic = edict({k: dict() for k in sample_fn_dic.keys()})

    global_attempts = 0
    global_success = 0

    pbar = tqdm(total=len(test_set)*opt.num_samples, desc="Sample")
    for epoch, item in enumerate(test_set):
        desc = 'Sample'
        print(f'epoch -> {epoch}/{len(test_set) - 1}, {item.id}')
        sampler = api.chem_sample(item, batch_size=opt.batch_size,
                                  assert_types=set(assert_types),
                                  assert_actions=set((ChemAssertActionEnum.LOGGING,)),
                                  max_attempts=opt.max_attempts,
                                  max_attempts_exceeded_action='stop',
                                  desc=item.id,
                                  use_groundtruth_rel=opt.groundtruth_rel,
                                  use_force_mean=opt.force_mean)
        res_lst = []
        for res in islice(sampler, opt.num_samples):
            res_lst.append(res)
            _global_attempts = global_attempts + sampler.num_attempts
            _global_success = global_success + sampler.num_success
            _global_succ = _global_success/_global_attempts*100 if _global_attempts > 0 else 0
            pbar.update(1)
            pbar.set_postfix(dict(
                epoch=epoch, global_succ=f'{_global_succ:.3f}%',
                **sampler.status_dic))
        if len(res_lst) < opt.num_samples:
            warnings.warn(f"Insufficient sampling quantity on {item.id}!")

        global_attempts += sampler.num_attempts
        global_success += sampler.num_success

        prd_smi_lst = [res.smi for res in res_lst]
        prd_frags_lst = [res.frags for res in res_lst]

        logging.info(pretty_kv(
            dict(id=item.id, **sampler.status_dic), ndigits=5, prefix=f'{desc}: '))

        sample_res_dic.status[item.id] = sampler.status_dic
        sample_res_dic.smi[item.id] = prd_smi_lst
        sample_res_dic.frags[item.id] = prd_frags_lst
        sample_res_dic.attn[item.id] = res_lst[0].attn

    pbar.close()

    for k, _fn in sample_fn_dic.items():
        auto_dump(sample_res_dic[k], _fn, json_indent=True)
        logging.info(f"Saved {k} -> {_fn}")

    logging.info("Done!")
