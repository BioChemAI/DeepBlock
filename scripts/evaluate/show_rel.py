import argparse
import hashlib
import logging
from pathlib import Path
import warnings
from easydict import EasyDict as edict
from typing import List, Tuple, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from deepblock.datasets import name_to_dataset_cls, \
    ComplexAADataset
from deepblock.utils.complex_to_aa import ComplexAAExtract
from deepblock.datasets import ComplexAAItem
from deepblock.utils import Toc, Vocab, auto_dump, auto_load, ifn, init_logging, \
    mix_config, sequential_file_hash, use_path, use_memory, rel_fn_dic

import pymol
from pymol import cmd

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_show_rel"

memory = use_memory()

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: QED, SA, Lipinski, LogP - CVAE Complex')

    parser.add_argument("--complex-id", type=str, required=True)
    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--docking-suffix", type=str)
    parser.add_argument("--dtb-work-dn", type=str, default="work/docking_toolbox")

    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--vocab-fn", type=str, default=f"saved/preprocess/merge_vocabs"
                        f"/chembl,crossdocked&frag_vocab.json")

    opt = mix_config(parser, None)
    return opt

class DockedFns:
    def __init__(self, work_dn: Path, job_id: str, complex_id: str, allow_partial=True) -> None:
        self.job_dn = use_path(dir_path=work_dn / 'jobs' / job_id, new=False)
        self.docked_dn = use_path(dir_path=self.job_dn / "docked", new=False)

        complex_lst: List[Dict[str, str]] = auto_load(self.job_dn / "complex.json")
        if allow_partial:
            complex_id_possible_lst = [x['id'] for x in complex_lst if x['id'].startswith(complex_id)]
            if len(complex_id_possible_lst) == 1:
                complex_id = complex_id_possible_lst[0]
            else:
                raise ValueError(f"Found {len(complex_id_possible_lst)} possible complex ID: "
                                 f"{str(complex_id_possible_lst)}")
        else:
            try:
                complex_id = next(x['id'] for x in complex_lst if x['id'] == complex_id)
            except StopIteration:
                raise ValueError(f"Complex ID {complex_id} not existed.")

        self.job_id = job_id
        self.complex_id = complex_id

        score_lst = auto_load(self.job_dn / "score.json")
        self.affinity = next((x['affinity'] for x in score_lst if x['id'] == complex_id), None)
        if self.affinity is None:
            warnings.warn("Docking score does not exist.")

        self.ligand_pdbqt = use_path(file_path=self.docked_dn / f'{complex_id}.docked_ligand.pdbqt', new=False)    # 1
        self.dock_log = use_path(file_path=self.docked_dn / f'{complex_id}.dock.log', new=False)                   # 2
        self.ligand_sdf = use_path(file_path=self.docked_dn / f'{complex_id}.docked_ligand.sdf', new=False)        # 3
        self.box_txt = use_path(file_path=self.docked_dn / f'{complex_id}.dock.box.txt', new=False)                # 4
        self.receptor_pdbqt = use_path(file_path=self.docked_dn / f'{complex_id}.receptor.pdbqt', new=False)       # 5

def find_hash_in_prd_lookup(dic: Dict, h: str):
    for cid, _v in dic.items():
        _v: Dict
        for smi, _h in _v.items():
            if _h == h:
                yield cid, smi

@memory.cache
def get_dataset_item(cid: str, opt: edict) -> Tuple[ComplexAAItem, edict, ComplexAAExtract]:
    # Dataset
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    meta_lst = _dataset.source("meta")
    id_to_meta = {meta.id: meta for meta in meta_lst} # To extract the reference SMILES
    x_vocab = Vocab(**auto_load(opt.vocab_fn)) if opt.vocab_fn else None
    _dataset_opt = dict(d=_dataset,
                        rel_fn=rel_fn_dic[opt.rel_fn_type],
                        x_vocab=x_vocab,
                        x_max_len=opt.x_max_len,
                        c_max_len=opt.c_max_len,
                        split_pro_dic=opt.split_pro,
                        is_dev=False)
    test_set = ComplexAADataset(**_dataset_opt, split_key='test')
    logging.info(f"len(test_set): {len(test_set)}")
    idx = next(i for i, x in enumerate(test_set.raw_lst) if x['id'] == cid)
    item = test_set[idx]
    meta = id_to_meta[cid]

    complex_to_aa_dic = _dataset.source("complex_to_aa")
    aa = complex_to_aa_dic[cid]
    return item, meta, aa

if __name__ == '__main__':
    opt = parse_opt()
    opt.docking_suffix = ifn(opt.docking_suffix, opt.suffix)

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    docking_input_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_input{opt.docking_suffix}.tar.gz"
    docking_lookup_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_lookup{opt.docking_suffix}.json"
    attn_fn = saved_dn / f"sample/attn{opt.suffix}.msgpack"
    opt_fn = use_path(file_path=saved_dn / "opt.json", new=False)
    
    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/{STAGE}{opt.suffix}.log")]
    
    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    ## Get complex_fns
    job_id = sequential_file_hash(docking_input_fn, hashlib.sha256()).hexdigest()

    docked_fns = DockedFns(work_dn=Path(opt.dtb_work_dn), 
                               job_id=job_id, complex_id=opt.complex_id, allow_partial=True)

    docking_lookup_dic = edict(auto_load(docking_lookup_fn))
    cid, smi = next(find_hash_in_prd_lookup(docking_lookup_dic['prd'], docked_fns.complex_id))
    logging.info(f"cid: {cid}, smi: {smi}")

    attn_dic = auto_load(attn_fn)
    attn = np.array(attn_dic[cid][0])
    attn = attn[1:-1] # Remove sos, eos

    bpt = auto_load(opt_fn, to_edict=True)
    item, meta, aa = get_dataset_item(cid, bpt)
    logging.info(f"item: {item}")
    logging.info(f"meta: {meta}")
    
    assert len(aa.ids) == len(item.c_rel)
    assert len(aa.ids) == len(attn)

    # What we get?
    # cid, smi, attn, item, aa, meta, docked_fns

    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    fontsize = 12

    fig, ax = plt.subplots(figsize=(6.5, 2.5), layout='constrained')
    ax.plot(aa.ids, item.c_rel, 'k.-', linewidth=1, label='实际')
    ax.plot(aa.ids, attn, 'kx-', linewidth=1, markersize=4, label='预测')
    ax.set_xlabel('2RMA Chain A 残基序列号', fontsize=fontsize)
    ax.set_ylabel('靶点贡献度', fontsize=fontsize)
    # ax.set_title("Attn")
    ax.legend(fontsize=fontsize)
    plt.savefig('tmp/靶点贡献度示意图.png', dpi=300, bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots(figsize=(6.5, 2.5), layout='constrained')
    # ax.plot(aa.ids, item.c_rel, 'k.-', linewidth=1, label='实际')
    # ax.plot(aa.ids, -attn, 'kx-', linewidth=1, label='预测')
    # ax.plot(aa.ids, np.zeros_like(aa.ids), 'k--', linewidth=1)
    # ax.set_xlabel('2RMA Chain A 残基序列号', fontsize=fontsize)
    # ax.set_ylabel('靶点贡献度', fontsize=fontsize)
    # # ax.set_title("Attn")
    # ax.legend(fontsize=fontsize)
    # plt.savefig('tmp/靶点贡献度示意图.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # ax.set_title("Attn")
    
    plt.savefig('tmp/靶点贡献度示意图.png', dpi=300, bbox_inches='tight')
    plt.show()

    exit()

    pymol.finish_launching(['pymol', '-W', '1280', '-H', '720'])
    cmd.set("seq_view", 1)

    # Load
    cmd.load(docked_fns.receptor_pdbqt, 'rec')
    cmd.load(docked_fns.ligand_pdbqt, 'lig')

    # Select active chain
    cmd.select("act_chain", f"chain {aa.id} in model rec")

    # What to show?
    what_to_show = "prd"
    # what_to_show = "ref"

    # Also dump a pymol script to set b value
    patch_lst = []

    # Put rel to b
    cmd.alter(f"model rec", f"b=0")
    patch_lst.append(f"alter model rec, b=0")

    for idx, resi in enumerate(aa.ids):
        b = attn[idx] if what_to_show == 'prd' else item.c_rel[idx]
        cmd.alter(f"resi {resi} in act_chain", f"b={b}")
        patch_lst.append(f"alter resi {resi} in act_chain, b={b}")

    with open(f"tmp/patch_{what_to_show}.pml", "w") as f:
        f.write('\n'.join(patch_lst))

    cmd.spectrum("b", "rainbow_rev", "rec")
    cmd.zoom("act_chain")
