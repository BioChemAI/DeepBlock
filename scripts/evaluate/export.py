import argparse
import logging
from pathlib import Path
from easydict import EasyDict as edict

from deepblock.utils import auto_dump, auto_load, auto_loadm, ifn, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_export"

ddof = 1

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Export - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--docking-suffix", type=str)
    parser.add_argument("--retro-db", type=str, default="work/retro_db")
    parser.add_argument("--export-fn", type=str, default=None)
    opt = mix_config(parser, None)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    opt.docking_suffix = ifn(opt.docking_suffix, opt.suffix)
    # Define Path
    opt.train_id = opt.base_train_id
    if opt.baseline:
        saved_dn = Path(f"saved/baseline/{opt.train_id}")
    else:
        saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    fn_dic = edict(
        sample_smi_fn = saved_dn / f"sample/smi{opt.suffix}.json",
        docking_score_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_score{opt.docking_suffix}.json",
        docking_lookup_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_lookup{opt.docking_suffix}.json",
        qedsa_fn = saved_dn / f"evalute{opt.evaluate_suffix}/qedsa{opt.suffix}.json",
        retro_db = Path(opt.retro_db) if "retro_db" in opt else None,
    )

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/export{opt.suffix}.log")]
    final_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/export{opt.suffix}.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Compute
    sample_smi_dic = auto_load(fn_dic.sample_smi_fn)
    qedsa_dic = auto_load(fn_dic.qedsa_fn)
    docking_score_dic = auto_load(fn_dic.docking_score_fn)
    docking_lookup_dic = auto_load(fn_dic.docking_lookup_fn)
    retro_dic = auto_loadm(sorted(fn_dic.retro_db.glob("*.done.json")))
    hash_to_score = {x['id']: x['affinity'] for x in docking_score_dic}

    final_dic = dict()
    for meta_id, smi_lst in sample_smi_dic.items():
        smi_to_score = dict()
        for smi in smi_lst:
            score_dic = dict()
            qedsa_item = qedsa_dic.get(smi, None)
            if qedsa_item is None:
                logging.warning(f"qedsa | SMILES not found: {meta_id}, {smi=}")
            score_dic.update(qedsa_item)

            h = docking_lookup_dic["prd"][meta_id].get(smi, None)
            vina_score = hash_to_score.get(h, None)
            if vina_score is None:
                logging.warning(f"vina_score | SMILES not found: {meta_id}, {smi=}")
            score_dic.update({"vina_score": vina_score})

            retro_item = retro_dic.get(smi, "UNVALIABLE")
            # if retro_item == "UNVALIABLE":
            #     logging.warning(f"retro | SMILES not found: {meta_id}, {smi=}")
            score_dic.update({"retro": retro_item})
            smi_to_score[smi] = score_dic
        final_dic[meta_id] = smi_to_score

    # logging.info(final_dic)
    auto_dump(final_dic, final_fn, json_indent=True)
    logging.info(f"Saved -> final_fn: {final_fn}")

    if opt.export_fn is not None:
        export_fn = use_path(file_path=opt.export_fn)
        auto_dump(final_dic, export_fn, json_indent=True)
        logging.info(f"Saved -> export_fn: {export_fn}")

    logging.info("Done!")
