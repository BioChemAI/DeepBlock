import argparse
import datetime
import logging
from pathlib import Path
import time

from deepblock.evaluation import DockingToolBox
from deepblock.utils import init_logging, mix_config, Toc, auto_dump


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Run batch docking')
    group = parser.add_argument_group('Docking Toolbox')
    DockingToolBox.argument_group(group)
    parser.add_argument("--input", type=str)
    parser.add_argument("--n-procs", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--output", type=str)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    input_fn_or_dn = Path(opt.input)
    log_fn = input_fn_or_dn.with_suffix(input_fn_or_dn.suffix+f'.run.{int(time.time()*1000)}.log')
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    dtb = DockingToolBox(**{k: opt[k] for k in DockingToolBox.opt_args if k in opt})
    bd = dtb.batch_dock(input_fn_or_dn=input_fn_or_dn, 
                        n_jobs=opt.n_jobs,
                        n_procs=opt.n_procs,
                        is_dev=opt.dev)
    
    toc = Toc()
    logging.info("Start -> run_jobs")
    score_lst = bd.run_jobs()
    logging.info(f"Finish -> run_jobs, toc: {datetime.timedelta(seconds=toc())}")


    logging.info("Start -> pack_tarball")
    tarball_fn = bd.pack_tarball(opt.output)
    logging.info(f"Finish -> pack_tarball, {tarball_fn}, toc: {datetime.timedelta(seconds=toc())}")

    if "input" in input_fn_or_dn.stem:
        score_fn = input_fn_or_dn.parent / (input_fn_or_dn.stem.replace('input', 'score') + '.json')
    else:
        score_fn = input_fn_or_dn.parent / (input_fn_or_dn.stem + '.score.json')

    auto_dump(score_lst, score_fn, json_indent=True)
    logging.info(f"Score list dumped to {score_fn}")

    logging.info("Done!")
