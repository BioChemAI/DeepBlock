import argparse
import logging
import datetime
from pathlib import Path
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from deepblock.datasets import ToxricDataset
from deepblock.utils import Toc, init_random, init_logging, use_path, \
    mix_config, pretty_kv, auto_dump
from deepblock.api import APIRegressTox

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "regress_tox"
STAGE = "chem_test"


def parse_opt():
    parser = argparse.ArgumentParser(description='Chem Test: Regress Tox')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--toxric-cached-dn", type=str,
                        default="saved/preprocess/toxric")
    parser.add_argument("--check-fingerprint", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20230402)
    parser.add_argument("--suffix", type=str)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    metric_res_fn = use_path(file_path=saved_dn / STAGE / f"metric_res.json")

    # Initialize log, config
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    _dataset = ToxricDataset(opt.toxric_cached_dn)
    train_set, test_set = _dataset\
        .download()\
        .preprocess(check_fp=opt.check_fingerprint)\
        .split(split_pro=opt.split_pro, ret_tup=("train", "test"))

    # API
    api = APIRegressTox(saved_dn)

    # Test
    toc = Toc()
    smi_lst = [x["smi"] for x in test_set]
    true_val_arr = np.array([x["val"] for x in test_set], dtype=np.float64)
    logging.info("Start!")
    pred_val_arr = np.array(api.chem_predict(smi_lst))
    metric_res_dic = {
        'RMSE': np.sqrt(mean_squared_error(true_val_arr, pred_val_arr)),
        'R2': r2_score(true_val_arr, pred_val_arr)
    }
    logging.info(f"metric_res: {pretty_kv(metric_res_dic, ndigits=5)}, "
                 f"toc: {datetime.timedelta(seconds=toc())}")

    # Save
    auto_dump(metric_res_dic, metric_res_fn)
    logging.info(f"Saved! metric_res -> {metric_res_fn}")

    logging.info("Done!")
