import argparse
import logging
from rdkit import Chem
from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)

from deepblock.utils import auto_dump, auto_load, auto_loadm, init_logging, \
    mix_config, use_path

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Retro Star - CVAE Complex')
    parser.add_argument("--db-dn", type=str, default='work/retro_db')
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    init_logging()

    db_dn = use_path(dir_path=opt.db_dn)
    out_fn = use_path(file_path=db_dn / "no_hs.done.json")
    done_fns = sorted(db_dn.glob('*.done.json'))
    done_dic = auto_loadm(done_fns)
    logging.info(f"Loading {len(done_dic)} results from {len(done_fns)} files")

    no_hs_dic = dict()
    for smi, data in tqdm(done_dic.items()):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            new_smi = Chem.MolToSmiles(mol)
            if isinstance(new_smi, str) and (new_smi not in done_dic) and (new_smi not in no_hs_dic):
                no_hs_dic[new_smi] = data

    auto_dump(no_hs_dic, out_fn, json_indent=True)
    logging.info(f"Saved {len(no_hs_dic)} results to {out_fn}")
    logging.info("Done!")
