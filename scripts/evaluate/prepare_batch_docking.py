import argparse
import logging
from pathlib import Path
import tempfile
from easydict import EasyDict as edict
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.datasets import name_to_dataset_cls, CrossDockedDataset, PDBbindDataset
from deepblock.evaluation import DockingToolBox
from deepblock.utils import Toc, auto_dump, auto_load, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "prepare_batch_docking"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Prepare batch docking - CVAE Complex')

    group = parser.add_argument_group('Docking Toolbox')
    DockingToolBox.argument_group(group)
    DockingToolBox.dock_kwargs_argument_group(group)

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--pdbbind-cached-dn", type=str,
                        default="saved/preprocess/pdbbind")
    parser.add_argument("--sbdd-dir", type=str)
    parser.add_argument("--dataset-dir", type=str)
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
    sample_smi_fn = use_path(file_path=saved_dn / f"sample/smi{opt.suffix}.json")

    opt.new_suffix = opt.suffix + f'_{opt.dock_backend}' if opt.dock_backend in DockingToolBox.DOCK_BACKEND_BUILTIN else "_other"
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute/batch_docking_input{opt.new_suffix}{'_dev' if opt.dev else ''}.log")]
    tarball_fn = use_path(file_path=saved_dn / f"evalute/batch_docking_input{opt.new_suffix}{'_dev' if opt.dev else ''}.auto")
    lookup_fn = use_path(file_path=saved_dn / f"evalute/batch_docking_lookup{opt.new_suffix}{'_dev' if opt.dev else ''}.json")
    tmp_dn = use_path(dir_path=opt.tmp_dn)
    dataset_dn = use_path(dir_path=opt.dataset_dir or opt.sbdd_dir, new=False)

    # Initialize log, device, config, wandb
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Dataset
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    meta_lst = _dataset.source("meta")
    if isinstance(_dataset, CrossDockedDataset):
        test_meta_lst = [meta for meta in meta_lst if meta.split == "test"]
    elif isinstance(_dataset, PDBbindDataset):
        test_id_set = _dataset.source("pick_set")["test"]
        test_meta_lst = [meta for meta in meta_lst if meta.id in test_id_set]
    else:
        raise ValueError(f"Unknown dataset: {opt.include}")
    if opt.dev:
        test_meta_lst = test_meta_lst[:3]
    logging.info(f"{len(test_meta_lst)=}, {test_meta_lst[0]}")

    # Docking Toolbox
    dtb = DockingToolBox(**{k: opt[k] for k in DockingToolBox.opt_args if k in opt})

    work_temporary_directory = tempfile.TemporaryDirectory(
        prefix=f'{PROJECT_NAME}-{MODEL_TYPE}-{STAGE}_',
        dir=tmp_dn)
    work_dn = Path(work_temporary_directory.name)

    toc = Toc()

    # Lookup Table
    sample_smi_dic = auto_load(sample_smi_fn)
    lookup_dic = edict(dict(ref={}, prd={}))

    # kwargs.json
    kwargs_dic = {k: opt[k] for k in DockingToolBox.dock_kwargs_opt_args if k in opt}
    auto_dump(kwargs_dic, work_dn / 'kwargs.json')

    # Batch Prepare
    logging.info('Start -> Batch Prepare (add)')
    bp = dtb.batch_prepare(work_dn)
    pbar = tqdm(test_meta_lst, desc='Batch Prepare (add)')
    for meta in pbar:
        complex = bp.add_complex(
            receptor=(bp.ReceptorTypeEnum.PDB_FN, dataset_dn / meta.protein),
            ligand=(bp.LigandTypeEnum.SDF_FN, dataset_dn / meta.ligand),
            box=(bp.BoxTypeEnum.SDF_FN, dataset_dn / meta.ligand),
        )
        lookup_dic.ref[meta.id] = complex.id

        prd_lookup_dic = dict()
        for smi in sample_smi_dic[meta.id]:
            complex = bp.add_complex(
                receptor=(bp.ReceptorTypeEnum.PDB_FN, dataset_dn / meta.protein),
                ligand=(bp.LigandTypeEnum.SMI_STR, smi),
                box=(bp.BoxTypeEnum.SDF_FN, dataset_dn / meta.ligand),
            )
            prd_lookup_dic[smi] = complex.id
        lookup_dic.prd[meta.id] = prd_lookup_dic
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
