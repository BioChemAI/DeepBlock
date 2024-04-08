import argparse
import logging
import time

from deepblock.utils import auto_dump, auto_load, auto_loadm, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_compute_retro_star"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Retro Star - CVAE Complex')
    parser.add_argument("--db-dn", type=str, default='work/retro_db')
    parser.add_argument("--glob-pattern", type=str, default='*.done.json')
    parser.add_argument("--out-fn", type=str)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    init_logging()

    db_dn = use_path(dir_path=opt.db_dn)
    if opt.out_fn is None:
        opt.out_fn = f"retro_db_merge_on_{int(time.time())}.done.json"
    out_fn = use_path(file_path=db_dn / opt.out_fn)
    backup_dn = use_path(dir_path=db_dn / "backup")
    done_fns = sorted(db_dn.glob(opt.glob_pattern))

    done_dic = auto_loadm(done_fns)
    logging.info(f"Loading {len(done_dic)} results from {len(done_fns)} files")
    auto_dump(done_dic, out_fn, json_indent=True)
    logging.info(f"Saved {len(done_dic)} results to {out_fn}")
    for fn in done_fns:
        fn.rename(backup_dn / fn.name)
    logging.info(f"Move {len(done_fns)} files to {backup_dn}")
    logging.info("Done!")
