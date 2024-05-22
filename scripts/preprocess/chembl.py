"""# Fragmenting the SMILES of the ChEMBL dataset

```bash
cd /dataset
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_31/chembl_31_chemreps.txt.gz
gzip -dk chembl_31_chemreps.txt.gz
```

Quoted from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_31/README

> chembl_31_chemreps.txt.gz            Tab-separated file containing different
>                                      chemical representations (SMILES, InChI
>                                      and InChI Key) of chembl_31 compounds,
>                                      includes chembl_id
"""

import argparse
import logging
from deepblock.datasets import ChEMBLDataset
from deepblock.utils import init_logging, mix_config, use_path

def parse_opt():
    parser = argparse.ArgumentParser(description='Preprocess: ChEMBL')
    parser.add_argument("--chembl-chemreps", type=str, default="/dataset/chembl_31_chemreps.txt")
    parser.add_argument("--cached-dn", type=str, default="saved/preprocess/chembl")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--frag-min-freq", type=int, default=100)
    parser.add_argument("--word-min-freq", type=int, default=80)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    init_logging(f"{opt.cached_dn}/preprocess.log")
    logging.info(opt)

    chembl_dataset = ChEMBLDataset(opt.cached_dn)
    chembl_dataset.preprocess(
        chembl_chemreps=use_path(file_path=opt.chembl_chemreps, new=False), 
        n_jobs=opt.n_jobs, is_dev=opt.dev,
        frag_min_freq=opt.frag_min_freq, 
        word_min_freq=opt.word_min_freq)

    logging.info("Done!")
