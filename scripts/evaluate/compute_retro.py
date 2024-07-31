import argparse
from contextlib import contextmanager
import logging
import os
import socket
from pathlib import Path
import random
import time
from typing import List, Dict

from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, use_path

from deepblock.evaluation import RetroStarPlanner

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_retro_star"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Retro Star - CVAE Complex')

    group = parser.add_argument_group('Retro Star Planner')
    RetroStarPlanner.argument_group(group)

    parser.add_argument("--db-dn", type=str, default='work/retro_db')
    parser.add_argument("--job-id", type=str)
    parser.add_argument("--job-id-suffix", type=str, default='')
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--smi-fn", type=str)
    parser.add_argument("--base-train-id", type=str)
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--enable-pend", action="store_true")
    parser.add_argument("--test-smi-suffix", type=str, default='')
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.job_id is None:
        opt.job_id = f'{socket.gethostname()}-{os.getpid()}-{int(time.time()*1000)}{opt.job_id_suffix}'

    # Define Path
    opt.train_id = opt.base_train_id

    ## Input
    if opt.smi_fn is None:
        assert opt.train_id is not None
        saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
        sample_smi_fn = saved_dn / f"sample/smi{opt.suffix}.json"
        ref_smi_fn = saved_dn / f"test_smi{opt.test_smi_suffix}.json"

        sample_smi_dic = auto_load(sample_smi_fn)
        ref_smi_dic = auto_load(ref_smi_fn)
        smi_set = set(smi for smi_lst in sample_smi_dic.values() for smi in smi_lst) | \
            set(ref_smi_dic.values())
    else:
        smi_set = set(auto_load(opt.smi_fn))

    ## Output
    db_dn = use_path(dir_path=opt.db_dn)
    log_fn = use_path(file_path=db_dn / f"{opt.job_id}.log")
    my_lock_fn = use_path(file_path=db_dn / f"{opt.job_id}.lock.json")
    my_pend_fn = use_path(file_path=db_dn / f"{opt.job_id}.pend.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    @contextmanager
    def pend_lock(my_pend_fn: Path, pend_dn: Path, enable: bool=True):
        if not enable:
            yield
            return
        
        get_pend_lst = lambda: {fn.relative_to(pend_dn).as_posix(): auto_load(fn) for fn in sorted(pend_dn.glob("*.pend.json"))}
        get_pend_queue = lambda: [x[0] for x in sorted(get_pend_lst().items(), key=lambda x: (x[1]["time"], x[1]["rand"]))]
        gen_pend_dic = lambda: {"time": time.time(), "rand": random.random()}

        my_pend_key = my_pend_fn.relative_to(pend_dn).as_posix()
        my_pend_dic = gen_pend_dic()
        auto_dump(my_pend_dic, my_pend_fn)
        time.sleep(5)

        try:
            while True:
                pend_queue: List[str] = get_pend_queue()
                logging.info(f"Pend: {my_pend_key} at Num. {pend_queue.index(my_pend_key)+1}/{len(pend_queue)}")
                if pend_queue[0] == my_pend_key:
                    yield
                    break
                else:
                    time.sleep(5)
        finally:
            my_pend_fn.unlink()

    with pend_lock(my_pend_fn, db_dn, enable=opt.enable_pend):
        # DB
        done_fns = sorted(db_dn.glob("*.done.json"))
        done_dic = {k: v for fn in done_fns for k, v in auto_load(fn).items()}
        logging.info(f"Loading {len(done_dic)} results from {len(done_fns)} files")

        lock_fns = sorted(db_dn.glob("*.lock.json"))
        lock_set = set(x for fn in lock_fns for x in auto_load(fn))
        logging.info(f"Loading {len(lock_set)} locks from {len(lock_fns)} files")

        # Filtering work
        filter_history_lst = []
        filter_history_lst.append(len(smi_set))
        smi_set = smi_set - lock_set - done_dic.keys()
        filter_history_lst.append(len(smi_set))
        smi_lst = list(smi_set)
        if opt.limit is not None:
            smi_lst = smi_lst[:opt.limit]
        filter_history_lst.append(len(smi_lst))
        logging.info(f"Fliter: {'>'.join(str(x) for x in filter_history_lst)}")

        # Lock
        auto_dump(smi_lst, my_lock_fn)

    # Retro Star Planner
    rsp = RetroStarPlanner(**{k: opt[k] for k in RetroStarPlanner.opt_args if k in opt})
    chunk_dic: Dict[str, Dict[str, float]] = dict()

    # Save
    num_chunk = 0
    def save_chunk():
        global num_chunk, chunk_dic
        toc = Toc()
        chunk_id = num_chunk
        num_chunk += 1
        chunk_fn = use_path(file_path=db_dn / f"{opt.job_id}.{chunk_id}.done.json")
        auto_dump(chunk_dic, chunk_fn, json_indent=True)
        logging.info(f"Saved -> chunk_fn: {chunk_fn}, toc: {toc():.3f}")
        chunk_dic.clear()

    for idx, smi in enumerate(smi_lst):
        toc = Toc()
        try:
            result = rsp(smi)
        except Exception as err:
            result = None
            logging.warning(f'({idx+1}/{len(smi_lst)}) {smi} -> error: {repr(err)}, toc: {toc():.3f}')
        else:
            logging.info(f'({idx+1}/{len(smi_lst)}) {smi} -> result: {result}, toc: {toc():.3f}')
            
        chunk_dic[smi] = result
        if (idx+1) % opt.save_every == 0:
            save_chunk()

    if len(chunk_dic) > 0:
        save_chunk()

    # Unlock
    try:
        my_lock_fn.unlink()
    except FileNotFoundError as err:
        logging.error(f"Unlock error: {repr(err)}")
    else:
        logging.info(f"Unlock success!")

    logging.info("Done!")
