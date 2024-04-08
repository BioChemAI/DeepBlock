import argparse
from itertools import count
import logging
import time

from typing import List

from deepblock.utils import auto_load, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_retro_star"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Retro Star - CVAE Complex')
    parser.add_argument("--db-dn", type=str, default='work/retro_db')
    parser.add_argument("--sleep", type=float)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    init_logging()

    db_dn = use_path(dir_path=opt.db_dn)
        
    get_pend_lst = lambda: {fn.relative_to(db_dn).as_posix(): auto_load(fn) for fn in sorted(db_dn.glob("*.pend.json"))}
    get_pend_queue = lambda: [x[0] for x in sorted(get_pend_lst().items(), key=lambda x: (x[1]["time"], x[1]["rand"]))]

    for epoch in count(1):
        logging.info(f"epoch: {epoch}")

        pend_queue: List[str] = get_pend_queue()
        logging.info(f"pend_queue: {pend_queue}")

        done_fns = sorted(db_dn.glob("*.done.json"))
        done_dic = {k: v for fn in done_fns for k, v in auto_load(fn).items()}
        logging.info(f"Loading {len(done_dic)} results from {len(done_fns)} files")

        lock_fns = sorted(db_dn.glob("*.lock.json"))
        lock_set = set(x for fn in lock_fns for x in auto_load(fn))
        logging.info(f"Loading {len(lock_set)} locks from {len(lock_fns)} files")

        done_set = set(done_dic.keys()) & lock_set
        logging.info(f"Progress: {len(done_set)}/{len(lock_set)}={len(done_set) / len(lock_set)*100 if len(lock_set) > 0 else 100:.3f}%")

        if opt.sleep is not None:
            time.sleep(opt.sleep)
        else:
            break

    logging.info("Done!")
