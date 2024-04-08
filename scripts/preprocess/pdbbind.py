"""# Fragmenting the SMILES of ligands and extracting amino acid sequence of proteins 
from PDBBind dataset.

Please download the following 3 files from the [PDBbind v2020](http://www.pdbbind.org.cn/download.php) 
to the same directory (assuming it is `$pdbbind_dir`).

1. Index files of PDBbind -> PDBbind_v2020_plain_text_index.tar.gz
2. Protein-ligand complexes: The general set minus refined set -> PDBbind_v2020_other_PL.tar.gz
3. Protein-ligand complexes: The refined set -> PDBbind_v2020_refined.tar.gz

To extract the files, first navigate to the `$pdbbind_dir` directory and then use the following command. 
*On a HDD, this could take a long time.*

```bash
tarballs=("PDBbind_v2020_plain_text_index.tar.gz" "PDBbind_v2020_refined.tar.gz" "PDBbind_v2020_other_PL.tar.gz")
for tarball in "${tarballs[@]}"
do
  dirname=${tarball%%.*}
  mkdir -p "$dirname" && pv -N "Extracting $tarball" "$tarball" | tar xzf - -C "$dirname"
done
```

The following files are required to exist:

- `$pdbbind_dir/PDBbind_v2020_plain_text_index/index/INDEX_general_PL.2020`
- `$pdbbind_dir/PDBbind_v2020_refined/refined-set/1a1e/1a1e_ligand.sdf`
- `$pdbbind_dir/PDBbind_v2020_other_PL/v2020-other-PL/1a0q/1a0q_protein.pdb`

"""

import argparse
import logging
from deepblock.datasets import PDBbindDataset
from deepblock.utils import init_logging, mix_config, use_path

def parse_opt():
    parser = argparse.ArgumentParser(description='Preprocess: PDBbind')
    parser.add_argument("--pdbbind-dir", type=str)
    parser.add_argument("--cached-dn", type=str, default="saved/preprocess/pdbbind")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--frag-min-freq", type=int, default=1)
    parser.add_argument("--word-min-freq", type=int, default=1)
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    init_logging(f"{opt.cached_dn}/preprocess.log")
    logging.info(opt)

    pdbbind_dataset = PDBbindDataset(opt.cached_dn)
    pdbbind_dataset.preprocess(
        pdbbind_dir=use_path(
            dir_path=opt.pdbbind_dir, new=False) if opt.pdbbind_dir else None, 
        n_jobs=opt.n_jobs, is_dev=opt.dev,
        frag_min_freq=opt.frag_min_freq, 
        word_min_freq=opt.word_min_freq)

    logging.info("Done!")
