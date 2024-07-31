import argparse
import logging
import pandas as pd
import numpy as np
from itertools import islice
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.utils import mix_config, init_random, init_logging, \
    use_path, auto_dump, complex_to_aa
from deepblock.api import APICVAEComplex, ITEM_MAKER_INPUT_TYPES, \
    ITEM_MAKER_GROUNDTRUTH_LIGAND_TYPES

CONSERVATIVE_SUCCESS_RATE = 0.5

def parse_opt():
    parser = argparse.ArgumentParser(description='Targeted molecular generation')
    parser.add_argument("--input-data", type=str, default='4IWQ')
    parser.add_argument("--input-type", type=str, default='pdb_id', 
                        choices=list(ITEM_MAKER_INPUT_TYPES))
    parser.add_argument("--input-chain-id", type=str, default=None, 
                        help=f"Chain ID for input data, special IDs: {list(complex_to_aa.CHAIN_SPECIAL_IDS)}")
    parser.add_argument("--output-json", type=str, default='tmp/generate_result.json')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230607)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--groundtruth-rel", action="store_true")
    parser.add_argument("--force-mean", action="store_true")
    parser.add_argument("--groundtruth-ligand-data", type=str)
    parser.add_argument("--groundtruth-ligand-type", type=str, default='sdf_fn', 
                        choices=list(ITEM_MAKER_GROUNDTRUTH_LIGAND_TYPES))
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    assert not (opt.force_mean and opt.groundtruth_rel), \
        "Args: --force-mean and --groundtruth-rel are mutually exclusive"
    conservative_max_attempts = int(np.ceil(opt.num_samples / CONSERVATIVE_SUCCESS_RATE))
    if opt.max_attempts is None:
        opt.max_attempts = conservative_max_attempts
    if opt.num_samples > opt.max_attempts:
        raise ValueError("max_attempts should be larger or equal to num_samples")

    # Output
    output_fn = use_path(file_path=opt.output_json)
    opt_fn = output_fn.with_suffix('.opt.json')
    log_fn = output_fn.with_suffix('.log')
    attn_fn = output_fn.with_suffix('.attn.msgpack')
    input_fn = output_fn.with_suffix('.rec.pdb')
    groundtruth_ligand_fn = output_fn.with_suffix('.groundtruth_lig.sdf')

    # Log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    auto_dump(opt, opt_fn, json_indent=True)
    logging.info(f'Exported options -> {opt_fn}')

    if opt.max_attempts < conservative_max_attempts:
        logging.warning("max_attempts is too small, may not generate enough samples")

    # API
    api = APICVAEComplex(device=opt.device)

    # Reproducibility
    init_random(opt.random_seed)

    maker = api.item_make(opt.input_data, opt.input_type, opt.input_chain_id,
                          opt.groundtruth_ligand_data, opt.groundtruth_ligand_type)
    logging.info(f'item: {maker.item}')
    if 'pdb' in maker.d:
        input_fn.write_bytes(maker.d['pdb'])
        logging.info(f'Exported input -> {input_fn}')
    if 'ligand_sdf' in maker.d:
        groundtruth_ligand_fn.write_bytes(maker.d['ligand_sdf'])
        logging.info(f'Exported groundtruth ligand -> {groundtruth_ligand_fn}')

    sampler = api.chem_sample(maker.item, batch_size=opt.batch_size,
                              max_attempts=opt.max_attempts,
                              desc=maker.item.id,
                              use_groundtruth_rel=opt.groundtruth_rel,
                              use_force_mean=opt.force_mean)

    pbar = tqdm(islice(sampler, opt.num_samples),
                total=opt.num_samples, desc="Generate")
    res_lst = []
    for res in pbar:
        res_lst.append(res)
        succ = sampler.num_success / sampler.num_attempts * 100
        pbar.set_postfix_str(f'succ: {succ:.3f}%, attempts: {sampler.num_attempts}')
    pbar.close()

    succ = sampler.num_success / sampler.num_attempts * 100
    logging.info(f'succ: {succ:.3f}%')

    evaluator = api.mol_evaluate(res.smi for res in res_lst)
    ind_lst = list(evaluator)

    output_obj = [{'smi': res.smi, 'frags': res.frags, 'ind': ind} for res, ind in zip(res_lst, ind_lst)]

    df = pd.DataFrame(output_obj)
    print(df)

    auto_dump(output_obj, output_fn, json_indent=True)
    logging.info(f'Exported output -> {output_fn}')

    attn_obj = dict(
        predict=res_lst[0].attn[0][1:-1].tolist() if res_lst[0].attn is not None else None,
        groundtruth=maker.item.c_rel.tolist() if maker.item.c_rel is not None else None)
    auto_dump(attn_obj, attn_fn)
    logging.info(f'Exported attention -> {attn_fn}')
