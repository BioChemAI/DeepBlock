import argparse
import logging
import pandas as pd
from itertools import islice
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.utils import mix_config, init_random, init_logging, \
    use_path, auto_dump
from deepblock.api import APICVAEComplex, ITEM_MAKER_INPUT_TYPES

def parse_opt():
    parser = argparse.ArgumentParser(description='Targeted molecular generation')
    parser.add_argument("--input-data", type=str, default='4IWQ')
    parser.add_argument("--input-type", type=str, default='pdb_id', 
                        choices=list(ITEM_MAKER_INPUT_TYPES))
    parser.add_argument("--output-json", type=str, default='tmp/generate_result.json')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=20230607)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=1<<12)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # Log
    init_logging()
    logging.info(f'opt: {opt}')

    # Output
    output_fn = use_path(file_path=opt.output_json)

    # API
    api = APICVAEComplex(device=opt.device)

    # Reproducibility
    init_random(opt.random_seed)

    maker = api.item_make(opt.input_data, opt.input_type)
    logging.info(f'item: {maker.item}')

    sampler = api.chem_sample(maker.item, batch_size=opt.batch_size,
                              max_attempts=opt.max_attempts,
                              desc=maker.item.id)

    pbar = tqdm(islice(sampler, opt.num_samples),
                total=opt.num_samples, desc="Generate")
    res_lst = list(pbar)
    pbar.close()

    succ = sampler.num_success / sampler.num_attempts * 100
    logging.info(f'succ: {succ:.3f}%')

    evaluator = api.mol_evaluate(res.smi for res in res_lst)
    ind_lst = list(evaluator)

    output_obj = [{'smi': res.smi, 'frags': res.frags, 'ind': ind} for res, ind in zip(res_lst, ind_lst)]

    df = pd.DataFrame(output_obj)
    print(df)

    auto_dump(output_obj, output_fn, json_indent=True)
    logging.info(f'Exported -> {output_fn}')
