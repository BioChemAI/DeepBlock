import argparse
from dataclasses import asdict
from itertools import groupby
import logging
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import esm

from deepblock.datasets import name_to_dataset_cls, \
    ComplexAADataset, ComplexAACollate
from deepblock.model import CVAEComplex
from deepblock.learner import LearnerCVAEComplex
from deepblock.utils import CheckpointManger, Vocab, \
    auto_dump, auto_load, generate_train_id, ifn, init_logging, \
    mix_config, init_random, pretty_kv, use_path, rel_fn_dic

PROJECT_NAME = "DeepBlock"

def parse_opt():
    parser = argparse.ArgumentParser(description='Train: CVAE Complex')
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--chembl-cached-dn", type=str,
                        default="saved/preprocess/chembl")
    parser.add_argument("--pdbbind-cached-dn", type=str,
                        default="saved/preprocess/pdbbind")
    parser.add_argument("--num-test-set", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20240723)
    parser.add_argument("--cached-dn", type=str, default="saved/preprocess/pdbbind")
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # Initialize log
    init_logging(f"{opt.cached_dn}/pick_set.log")
    logging.info(opt)

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset crossdocked
    _dataset = name_to_dataset_cls["crossdocked"](opt["crossdocked_cached_dn"])
    _dataset_opt = dict(d=_dataset, split_pro_dic={"train": 1.0})
    crossdocked_test_set = ComplexAADataset(**_dataset_opt, split_key='test')
    crossdocked_train_set = ComplexAADataset(**_dataset_opt, split_key='train')
    crossdocked_full_set = ComplexAADataset(**_dataset_opt)

    crossdocked_test_c_set = set([x["c_seq"] for x in crossdocked_test_set.raw_lst])
    crossdocked_train_c_set = set([x["c_seq"] for x in crossdocked_train_set.raw_lst])
    crossdocked_full_c_set = set([x["c_seq"] for x in crossdocked_full_set.raw_lst])

    logging.info(f"{len(crossdocked_test_set)=}, {len(crossdocked_train_set)=}")
    logging.info(f"{len(crossdocked_test_c_set)=}, {len(crossdocked_train_c_set)=}")
    logging.info(f"{len(crossdocked_test_c_set & crossdocked_train_c_set)=}")

    # Dataset
    _dataset = name_to_dataset_cls["pdbbind"](opt["pdbbind_cached_dn"])
    _dataset_opt = dict(d=_dataset)
    pdbbind_full_set = ComplexAADataset(**_dataset_opt)
    logging.info(f"{len(pdbbind_full_set)=}, {len(pdbbind_full_set)=}")

    pdbbind_full_c_set = set([x["c_seq"] for x in pdbbind_full_set.raw_lst])

    logging.info(f"{len(pdbbind_full_c_set & crossdocked_full_c_set)=}")
    logging.info(f"{len(pdbbind_full_c_set)=}, {len(crossdocked_full_c_set)=}")

    # Pick test set
    full_c_set = pdbbind_full_c_set - crossdocked_test_c_set
    test_c_set = set(random.sample(sorted(full_c_set), opt.num_test_set))
    train_c_set = full_c_set - test_c_set

    test_raw_lst = []
    train_raw_lst = []

    keyfunc = lambda x: x["c_seq"]
    data = sorted(pdbbind_full_set.raw_lst, key=keyfunc)
    for k, g in groupby(data, keyfunc):
        g = list(g)
        if k in test_c_set:
            test_raw_lst.append(random.choice(g))
        if k in train_c_set:
            train_raw_lst.extend(g)

    test_id_lst = [x["id"] for x in test_raw_lst]
    train_id_lst = [x["id"] for x in train_raw_lst]

    assert len(set(test_id_lst)) == len(test_id_lst)
    assert len(set(train_id_lst)) == len(train_id_lst)
    assert len(set(test_id_lst) & set(train_id_lst)) == 0

    logging.info(f"{len(test_id_lst)=}, {len(train_id_lst)=}")
    logging.info(f"{len(pdbbind_full_c_set)=}, {len(pdbbind_full_set)=}")

    logging.info(f"{test_id_lst=}")

    pick_set_fn = f"{opt.cached_dn}/pick_set.json"
    auto_dump(dict(test=test_id_lst, train=train_id_lst), pick_set_fn)
    logging.info(f"Saved! {pick_set_fn}")
