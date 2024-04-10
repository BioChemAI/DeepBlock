# DeepBlock

![GitHub license](https://img.shields.io/github/license/BioChemAI/DeepBlock.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/BioChemAI/DeepBlock.svg)
![GitHub contributors](https://img.shields.io/github/contributors/BioChemAI/DeepBlock)
![GitHub forks](https://badgen.net/github/forks/BioChemAI/DeepBlock)
![GitHub stars](https://img.shields.io/github/stars/BioChemAI/DeepBlock.svg)

This is the official implantation of the paper "A Deep Learning Approach for Rational Ligand Generation with Property Control via Reactive Building Blocks".
Additionally, we offer a user-friendly [web server](https://biochemai.app.pizyds.com/) to implement the functionality of DeepBlock.

## Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Develop](#Develop)
    
    - [Peprocess](#Peprocess)
   
    - [Train](#Train)
    
    - [Inference](#Inference)


## Installation

```bash
git clone git@github.com:BioChemAI/DeepBlock.git
cd deepblock
conda env create -f environment.yml
conda activate deepblock_env
pip install -e .
```

Also, for Docker

```bash
git clone git@github.com:BioChemAI/DeepBlock.git
cd deepblock
docker build --target base -t deepblock .
docker run -it --rm deepblock
```

## Usage

Quick start, `--input-type` can be 'seq', 'pdb', 'url', 'pdb_fn', 'pdb_id'.

```bash
python scripts/quick_start/generate.py \
    --input-data 4IWQ \
    --input-type pdb_id \
    --num-samples 16 \
    --output-json tmp/generate_result.json
```

## Develop

It is recommended to use VSCode for development, as debugging configuration files are already available in `.vscode`.

### Peprocess dataset

#### ChEMBL Dataset

```bash
python scripts/preprocess/chembl.py \
    --chembl-chemreps <PATH TO chembl_31_chemreps.txt>
```

#### CrossDocked Dataset (Index by [3D-Generative-SBDD](https://github.com/luost26/3D-Generative-SBDD))

```bash
python scripts/preprocess/crossdocked.py \
    --sbdd-dir <PATH TO crossdocked_pocket10_with_protein>
```

#### PDBbind Dataset (Optional)

```bash
python scripts/preprocess/pdbbind.py \
    --pdbbind-dir <PATH TO PDBBind>
```

#### Merge Dictionary

```bash
python scripts/preprocess/merge_vocabs.py \
    --includes chembl crossdocked
```

### Train

#### Pretraining on ChEMBL Dataset with ChEMBL+CrossDocked Dictionary

```bash
python scripts/cvae_complex/train.py \
    --include chembl \
    --device cuda:0 \
    --config scripts/cvae_complex/frag_pretrain_config.yaml \
    --vocab-fn "saved/preprocess/merge_vocabs/chembl,pdbbind&frag_vocab.json" \
    --no-valid-prior
```

#### Training on CrossDocked Dataset

Replace `20230303_191022_be9e` with pretrain ID.

```bash
python scripts/cvae_complex/train.py \
    --include crossdocked \
    --device cuda:0 \
    --config scripts/cvae_complex/complex_config.yaml \
    --base-train-id 20230303_191022_be9e
```

### Inference

#### Ligand Generation

Replace `20230305_163841_cee4` with train ID.

```bash
python scripts/cvae_complex/sample.py \
    --include crossdocked \
    --device cuda:0 \
    --base-train-id 20230305_163841_cee4 \
    --num-samples 100 \
    --validate-mol \
    --embed-mol \
    --unique-mol
```

#### Affinity optimization

```bash
python scripts/cvae_complex/optimize.py \
    --device cpu \
    --base-train-id 20230305_163841_cee4 \
    --num-samples 5000 \
    --complex-id F16P1_HUMAN_1_338_0/3kc1_A_rec_3kc1_2t6_lig_tt_min_0
```

#### Property optimization (SA)

```bash
python scripts/cvae_complex/sample_sa.py \
    --device cpu \
    --base-train-id 20230305_163841_cee4 \
    --num-samples 100 \
    --num-steps 50 \
    --complex-id F16P1_HUMAN_1_338_0/3kc1_A_rec_3kc1_2t6_lig_tt_min_0
```
