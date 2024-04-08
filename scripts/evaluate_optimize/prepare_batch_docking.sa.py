import argparse
from itertools import chain
import logging
from pathlib import Path
import tempfile
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import name_to_dataset_cls
from deepblock.evaluation import DockingToolBox
from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "prepare_batch_docking.optimize.sa"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Prepare batch docking for optimize - CVAE Complex')

    group = parser.add_argument_group('Docking Toolbox')
    DockingToolBox.argument_group(group)
    DockingToolBox.dock_kwargs_argument_group(group)

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--sbdd-dir", type=str)
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--tmp-dn", type=str, default="tmp")
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.suffix is None:
        opt.suffix = ''
    if opt.n_jobs == 0:
        raise ValueError("n_jobs == 0 in Parallel has no meaning")

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    info_fn = use_path(file_path=saved_dn / f"sample_sa/info{opt.suffix}.json")

    opt.new_suffix = opt.suffix + f'_{opt.dock_backend}' if opt.dock_backend in DockingToolBox.DOCK_BACKEND_BUILTIN else "_other"
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"sample_sa/batch_docking_input{opt.new_suffix}.log")]
    tarball_fn = use_path(file_path=saved_dn / f"sample_sa/batch_docking_input{opt.new_suffix}.auto")
    lookup_fn = use_path(file_path=saved_dn / f"sample_sa/batch_docking_lookup{opt.new_suffix}.json")
    tmp_dn = use_path(dir_path=opt.tmp_dn)
    sbdd_dn = use_path(dir_path=opt.sbdd_dir, new=False)

    # Initialize log, device, config, wandb
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Dataset
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    meta_lst = _dataset.source("meta")
    test_meta_lst = [meta for meta in meta_lst if meta.split == "test"]

    # Docking Toolbox
    dtb = DockingToolBox(**{k: opt[k] for k in DockingToolBox.opt_args if k in opt})

    work_temporary_directory = tempfile.TemporaryDirectory(
        prefix=f'{PROJECT_NAME}-{MODEL_TYPE}-{STAGE}_',
        dir=tmp_dn)
    work_dn = Path(work_temporary_directory.name)

    toc = Toc()

    # Lookup Table
    info_dic = auto_load(info_fn)
    smi_all_lst = list(chain.from_iterable(info_dic['popu']))
    smi_lst = list(set(smi_all_lst))
    logging.info(f"Total: {len(smi_all_lst)} -> {len(smi_lst)}")
    lookup_dic = dict()

    # kwargs.json
    kwargs_dic = {k: opt[k] for k in DockingToolBox.dock_kwargs_opt_args if k in opt}
    auto_dump(kwargs_dic, work_dn / 'kwargs.json')

    id_to_test_meta = {meta.id: meta for meta in test_meta_lst}
    meta = id_to_test_meta[info_dic['opt']['complex_id']]

    # Batch Prepare
    logging.info('Start -> Batch Prepare (add)')
    bp = dtb.batch_prepare(work_dn)

    # Ref
    complex = bp.add_complex(
        receptor=(bp.ReceptorTypeEnum.PDB_FN, sbdd_dn / meta.protein),
        ligand=(bp.LigandTypeEnum.SDF_FN, sbdd_dn / meta.ligand),
        box=(bp.BoxTypeEnum.SDF_FN, sbdd_dn / meta.ligand),
    )
    lookup_dic["!ref"] = complex.id

    # Prd
    pbar = tqdm(smi_lst, desc='Batch Prepare (add)')
    for smi in pbar:
        complex = bp.add_complex(
            receptor=(bp.ReceptorTypeEnum.PDB_FN, sbdd_dn / meta.protein),
            ligand=(bp.LigandTypeEnum.SMI_STR, smi),
            box=(bp.BoxTypeEnum.SDF_FN, sbdd_dn / meta.ligand),
        )
        lookup_dic[smi] = complex.id
        pbar.set_postfix({
            "len(delayed_jobs)": len(bp.delayed_jobs), 
            "len(complex_lst)": len(bp.complex_lst)})

    auto_dump(lookup_dic, lookup_fn, json_indent=True)
    auto_dump(bp.complex_lst, work_dn / 'complex.json', json_indent=True)
    logging.info(f'Finsh -> Batch Prepare (add), toc: {toc():.3f}\n{lookup_fn}\n{work_dn / "complex.json"}')

    logging.info(f'Start -> Batch Prepare (run), len(delayed_jobs): {len(bp.delayed_jobs)}')
    bp.run_jobs(n_jobs=opt.n_jobs)
    logging.info(f'Finsh -> Batch Prepare (run), toc: {toc():.3f}')

    logging.info(f'Start -> Batch Prepare (pack)')
    tarball_fn = bp.pack_tarball(tarball_fn)
    logging.info(f'Finsh -> Batch Prepare (pack), toc: {toc():.3f}, tarball_fn: {tarball_fn}')

    work_temporary_directory.cleanup()
    logging.info("Done!")
