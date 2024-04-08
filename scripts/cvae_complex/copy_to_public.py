"""Copy to public for CVAE Complex Model
"""

import argparse
import logging
import shutil
from pathlib import Path
from deepblock.utils import init_logging, mix_config, use_path, \
    CheckpointManger
from deepblock.public import PUBLIC_DN

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "copy_to_public"

def parse_opt():
    parser = argparse.ArgumentParser(description='Copy to public: CVAE Complex')
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--base-weight-choice", type=str, default="latest")
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    # Initialize log
    init_logging()
    logging.info(f'opt: {opt}')
    
    opt_fn = use_path(file_path=saved_dn / "opt.json", new=False)
    weights_dn = use_path(dir_path=saved_dn / "weights", new=False)
    best_fn = use_path(file_path=saved_dn / "best.json", new=False)
    ckpt = CheckpointManger(weights_dn, best_fn, save_step=10)
    weight_fn = ckpt.pick(opt.base_weight_choice)
    vocab_fn = use_path(dir_path=saved_dn / "x_vocab.json", new=False)

    required_fn_lst = [opt_fn, best_fn, weight_fn, vocab_fn]

    for src_fn in required_fn_lst:
        trg_fn = PUBLIC_DN / src_fn
        use_path(file_path=trg_fn)
        shutil.copy(src_fn, trg_fn)
        logging.info(f'Copy {src_fn} -> {trg_fn}')

    logging.info("Done!")
