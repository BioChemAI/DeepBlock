import argparse
import logging
import datetime
from pathlib import Path

from deepblock.datasets import ToxricDataset
from deepblock.utils import Toc, init_random, init_logging, use_path, \
    generate_train_id, mix_config, auto_dump
from deepblock.api import APIRegressTox

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "regress_tox"
STAGE = "train"


def parse_opt():
    parser = argparse.ArgumentParser(description='Train: Regress Tox')
    parser.add_argument("--toxric-cached-dn", type=str,
                        default="saved/preprocess/toxric")
    parser.add_argument("--check-fingerprint", action="store_true")
    parser.add_argument("--tpot-scale", choices=APIRegressTox.tpot_scales, default="full")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--suffix", type=str)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.dev: opt.tpot_scale = "tiny"

    # Define Path
    opt.train_id = generate_train_id(opt.dev, opt.suffix)
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    opt_fn = use_path(file_path=saved_dn / "opt.json")

    # Initialize log, config
    init_logging(log_fn)
    logging.info(f'opt: {opt}')
    auto_dump(dict(opt), opt_fn, json_indent=True)

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    _dataset = ToxricDataset(opt.toxric_cached_dn)
    train_set, test_set = _dataset\
        .download()\
        .preprocess(check_fp=opt.check_fingerprint)\
        .split(split_pro=opt.split_pro, ret_tup=("train", "test"))

    # API
    api = APIRegressTox(saved_dn, training=True)

    # Train
    toc = Toc()
    logging.info("Start!")
    test_score = api.train(train_set, test_set, scale=opt.tpot_scale, n_jobs=opt.n_jobs)
    logging.info(f"test_score: {test_score:.5f}, toc: {datetime.timedelta(seconds=toc())}")

    # Save
    for k, v in api.fn_dic.items():
        logging.info(f"Saved! {k} -> {v}")

    logging.info("Done!")
