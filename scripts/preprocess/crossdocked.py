"""# Fragmenting the SMILES of ligands and extracting amino acid sequence of proteins 
from CrossDocked dataset.

Reading the 3D-Generative-SBDD's index file (filter_index.pkl), index ligand and pocket 
from SBDD's directory (crossdocked_pocket10), copy protein file from CrossDocked2020 to 
SBDD's directory. Process all proteins and ligands.


The following files are required to exist:

- `$sbdd_dir/split_by_name.pt`
- `$sbdd_dir/index.pkl`
- `$sbdd_dir/1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0_pocket10.pdb`
- `$crossdocked_dir/1B57_HUMAN_25_300_0/5u98_D_rec.pdb`

"""

import argparse
import logging
from deepblock.datasets import CrossDockedDataset
from deepblock.utils import init_logging, mix_config, use_path

def parse_opt():
    parser = argparse.ArgumentParser(description='Preprocess: CrossDocked')
    parser.add_argument("--sbdd-dir", type=str)
    parser.add_argument("--crossdocked-dir", type=str, default=None)
    parser.add_argument("--cached-dn", type=str, default="saved/preprocess/crossdocked")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--frag-min-freq", type=int, default=1)
    parser.add_argument("--word-min-freq", type=int, default=1)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    init_logging(f"{opt.cached_dn}/preprocess.log")
    logging.info(opt)

    crossdocked_dataset = CrossDockedDataset(opt.cached_dn)
    crossdocked_dataset.preprocess(
        sbdd_dir=use_path(dir_path=opt.sbdd_dir, new=False), 
        crossdocked_dir=use_path(
            dir_path=opt.crossdocked_dir, new=False) if opt.crossdocked_dir else None, 
        n_jobs=opt.n_jobs, is_dev=opt.dev,
        frag_min_freq=opt.frag_min_freq, 
        word_min_freq=opt.word_min_freq)

    logging.info("Done!")
