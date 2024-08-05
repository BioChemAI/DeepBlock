import argparse
import logging
from numbers import Number
from pathlib import Path
from easydict import EasyDict as edict
from typing import Tuple, Dict, Union
import numpy as np

from deepblock.utils import Toc, auto_dump, auto_load, auto_loadm, ifn, init_logging, \
    mix_config, use_path

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "evaluate_summary"

ddof = 1

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Summary - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--evaluate-suffix", type=str, default='')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--docking-suffix", type=str)
    parser.add_argument("--retro-db", type=str, default="work/retro_db")
    parser.add_argument("--test-smi-suffix", type=str, default='')
    parser.add_argument("--ign-suffix", type=str)
    parser.add_argument("--export-fn", type=str, default=None)
    opt = mix_config(parser, None)
    return opt

class SummaryPipline:

    def __init__(self, **fns) -> None:
        self.fns = edict(fns)
        self.pipline = {
            "Count": self.item_count,
            "Vina Affinity": self.item_vina,
            "QED, SA, Lipinski, LogP": self.item_qedsa,
            "Retro* Sucess Rate": self.item_retro,
            "Distribution": self.item_dist,
            "IGN Affinity": self.item_ign,
        }

        # Final Dict format like {labels: value, ...}
        # ('Vina Affinity, 'mean'): -7.13347485657237
        self.final_dic: Dict[Tuple[str, ...], Union[Number, str]] = dict()

    def compute(self):
        for i, (k, v) in enumerate(self.pipline.items()):
            logging.info(f"Compute -> {i}. {k}")
            v(k)
    
    @staticmethod
    def append_mms_to_dic(dic, arr, *names):
        for mms, value in zip(
                ('mean', 'med', 'std'), 
                (np.mean(arr), np.median(arr), np.std(arr, ddof=ddof))):
            dic[(*names, mms)] = float(value)

    # 0. Count
    def item_count(self, name):
        try:
            sample_smi_dic = auto_load(self.fns.sample_smi_fn)                   # {id: [smi, ...], ...}
            sample_ref_smi_dic = auto_load(self.fns.sample_ref_smi_fn)           # {id: smi, ...}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            cid_to = edict()
            cid_to.count: Dict[str, int] = dict()
            for cid, smi_lst in sample_smi_dic.items():
                cid_to.count[cid] = len(smi_lst)
            self.final_dic[(name, 'sum')] = sum(cid_to.count.values())
            self.final_dic[(name, 'ref', 'sum')] = len(sample_ref_smi_dic)

    # 1. Vina Affinity
    def item_vina(self, name):
        try:
            sample_smi_dic = auto_load(self.fns.sample_smi_fn)                   # {id: [smi, ...], ...}
            sample_ref_smi_dic = auto_load(self.fns.sample_ref_smi_fn)           # {id: smi, ...}
            docking_score_dic = auto_load(self.fns.docking_score_fn)             # [{id(hash), score}, ...]      Data will be missing!!
            docking_lookup_dic = edict(auto_load(self.fns.docking_lookup_fn))    # {ref: {id: hash, ...}, prd: {id: {smi: hash, ...}, ...}}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            hash_to_score = {x['id']: x['affinity'] for x in docking_score_dic if x['affinity'] < 0}
            # hash_to_score = {x['id']: x['affinity'] for x in docking_score_dic}
            cid_to = edict()
            cid_to.mean: Dict[str, float] = dict()
            cid_to.miss: Dict[str, int] = dict()
            cid_to.count: Dict[str, int] = dict()
            cid_to.high: Dict[str, float] = dict()

            ref_docking_succ_cid_lst = [cid for cid in sample_smi_dic.keys() if docking_lookup_dic.ref[cid] in hash_to_score]
            if len(sample_smi_dic) - len(ref_docking_succ_cid_lst) > 0:
                logging.warning(f"Ref Docking Fail: {len(sample_smi_dic) - len(ref_docking_succ_cid_lst)}")
            for cid in ref_docking_succ_cid_lst:
                smi_lst = sample_smi_dic[cid]
                smi_to_hash = docking_lookup_dic.prd[cid]
                h_lst = [smi_to_hash[smi] for smi in smi_lst]
                score_lst = [hash_to_score[h] for h in h_lst if h in hash_to_score]
                cid_to.count[cid] = len(score_lst)
                cid_to.mean[cid] = np.mean(score_lst)
                cid_to.miss[cid] = len(h_lst) - len(score_lst)
                # Be Careful lt means high
                cid_to.high[cid] = np.mean(np.array(score_lst) <= hash_to_score[docking_lookup_dic.ref[cid]]) * 100

            mean_arr = np.array(list(cid_to.mean.values()))
            self.append_mms_to_dic(self.final_dic, mean_arr, name)
            mean_arr = np.array(list(cid_to.high.values()))
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'high', 'scale100')
            self.final_dic[(name, 'miss', 'sum')] = sum(cid_to.miss.values())

            h_lst = [docking_lookup_dic.ref[cid] for cid in sample_ref_smi_dic.keys()]
            score_lst = [hash_to_score[h] for h in h_lst if h in hash_to_score]
            if len(h_lst) - len(score_lst) > 0:
                logging.warning(f"Ref Docking Fail: {len(h_lst) - len(score_lst)}")
            mean_arr = np.array(score_lst)
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'ref')

    # 2. QED, SA, Lipinski, LogP
    def item_qedsa(self, name):
        try:
            sample_smi_dic = auto_load(self.fns.sample_smi_fn)                   # {id: [smi, ...], ...}
            sample_ref_smi_dic = auto_load(self.fns.sample_ref_smi_fn)           # {id: smi, ...}
            qedsa_dic = auto_load(self.fns.qedsa_fn)                             # {smi: {qed, sa, lipinski, logp, logp_1, ...}, ...}    Data will be missing!!
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            cid_to = edict()
            better_names = ('QED', 'SA', 'Lipinski', 'LogP', *(f'Lipinski {i+1}' for i in range(5)))
            key_names = ('qed', 'sa', 'lipinski', 'logp', *(f'lipinski_{i+1}' for i in range(5)))
            for bn in better_names:
                cid_to[bn]: Dict[str, float] = dict()
            for bn, kn in zip(better_names, key_names):
                for cid, smi_lst in sample_smi_dic.items():
                    value_lst = [qedsa_dic[smi][kn] for smi in smi_lst if smi in qedsa_dic]
                    value_lst = [v for v in value_lst if v is not None]
                    cid_to[bn][cid] = np.mean(value_lst)

                mean_arr = np.array(list(cid_to[bn].values()))
                self.append_mms_to_dic(self.final_dic, mean_arr, bn)

                value_lst = [qedsa_dic[smi][kn] for smi in sample_ref_smi_dic.values()]
                mean_arr = np.array(value_lst)
                self.append_mms_to_dic(self.final_dic, mean_arr, bn, 'ref')

    # 3. Distribution
    def item_dist(self, name):
        try:
            dist_dic = auto_load(self.fns.dist_fn)                               # {id: smi, ...}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            key_to_better_name = {
                "similarity": ("Diversity",),
                "ref_similarity": ("Similarity to Ground Truth",),
                "novelty_fp": ("Novelty by Fingerpoints", "scale100",),
                "scaffold_similarity": ("Scaffold Diversity",),
                "scaffold_ref_similarity": ("Scaffold Similarity to Ground Truth",),
                "scaffold_novelty_fp": ("Scaffold Novelty by Fingerpoints", "scale100",),
                "novelty_smi": ("Novelty by SMILES", "scale100",)
            }
            neg_key_set = {
                "similarity",
                "scaffold_similarity",
            }
            scale100_key_set = {
                "novelty_fp",
                "scaffold_novelty_fp",
                "novelty_smi",
            }
            assert len(neg_key_set - set(key_to_better_name.keys())) == 0
            assert len(scale100_key_set - set(key_to_better_name.keys())) == 0

            for key, bn in key_to_better_name.items():
                mean_arr = np.array([x[key] for x in dist_dic.values()])
                if key in neg_key_set:
                    mean_arr = 1 - mean_arr
                if key in scale100_key_set:
                    mean_arr = mean_arr * 100
                self.append_mms_to_dic(self.final_dic, mean_arr, *bn)

    # 4. Retro
    def item_retro(self, name):
        try:
            sample_smi_dic = auto_load(self.fns.sample_smi_fn)                   # {id: [smi, ...], ...}
            sample_ref_smi_dic = auto_load(self.fns.sample_ref_smi_fn)           # {id: smi, ...}
            if self.fns.retro_db.is_dir():
                retro_dic = auto_loadm(sorted(self.fns.retro_db.glob("*.done.json")))
            else:
                retro_dic = auto_load(self.fns.retro_db)                         # {smi: None or {}, ...}    Data cannot missing!!
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            cid_to_retro = dict()
            cid_to_miss = dict()
            for cid, smi_lst in sample_smi_dic.items():
                retro_ok_lst = [retro_dic[smi] is not None for smi in smi_lst if smi in retro_dic]
                cid_to_miss[cid] = len(smi_lst) - len(retro_ok_lst)
                cid_to_retro[cid] = np.mean(retro_ok_lst)

            self.final_dic[(name, 'miss', 'sum')] = sum(cid_to_miss.values())

            mean_arr = np.array(list(cid_to_retro.values())) * 100
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'scale100')

            mean_arr = np.array([retro_dic[smi] is not None for smi in sample_ref_smi_dic.values()]) * 100
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'ref', 'scale100')


    # 5. IGN Affinity
    def item_ign(self, name):
        try:
            sample_smi_dic = auto_load(self.fns.sample_smi_fn)                   # {id: [smi, ...], ...}
            sample_ref_smi_dic = auto_load(self.fns.sample_ref_smi_fn)           # {id: smi, ...}
            docking_score_dic = auto_load(self.fns.ign_score_fn)             # [{id(hash), score}, ...]      Data will be missing!!
            docking_lookup_dic = edict(auto_load(self.fns.docking_lookup_fn))    # {ref: {id: hash, ...}, prd: {id: {smi: hash, ...}, ...}}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            hash_to_score = docking_score_dic
            cid_to = edict()
            cid_to.mean: Dict[str, float] = dict()
            cid_to.miss: Dict[str, int] = dict()
            cid_to.count: Dict[str, int] = dict()
            cid_to.high: Dict[str, float] = dict()

            ref_docking_succ_cid_lst = [cid for cid in sample_smi_dic.keys() if docking_lookup_dic.ref[cid] in hash_to_score]
            if len(sample_smi_dic) - len(ref_docking_succ_cid_lst) > 0:
                logging.warning(f"Ref Docking Fail: {len(sample_smi_dic) - len(ref_docking_succ_cid_lst)}")
            for cid in ref_docking_succ_cid_lst:
                smi_lst = sample_smi_dic[cid]
                smi_to_hash = docking_lookup_dic.prd[cid]
                h_lst = [smi_to_hash[smi] for smi in smi_lst]
                score_lst = [hash_to_score[h] for h in h_lst if h in hash_to_score]
                cid_to.count[cid] = len(score_lst)
                cid_to.mean[cid] = np.mean(score_lst)
                cid_to.miss[cid] = len(h_lst) - len(score_lst)
                # Be Careful lt means high
                cid_to.high[cid] = np.mean(np.array(score_lst) <= hash_to_score[docking_lookup_dic.ref[cid]]) * 100

            mean_arr = np.array(list(cid_to.mean.values()))
            self.append_mms_to_dic(self.final_dic, mean_arr, name)
            mean_arr = np.array(list(cid_to.high.values()))
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'high', 'scale100')
            self.final_dic[(name, 'miss', 'sum')] = sum(cid_to.miss.values())

            h_lst = [docking_lookup_dic.ref[cid] for cid in sample_ref_smi_dic.keys()]
            score_lst = [hash_to_score[h] for h in h_lst if h in hash_to_score]
            if len(h_lst) - len(score_lst) > 0:
                logging.warning(f"Ref Docking Fail: {len(h_lst) - len(score_lst)}")
            mean_arr = np.array(score_lst)
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'ref')


if __name__ == '__main__':
    opt = parse_opt()
    opt.docking_suffix = ifn(opt.docking_suffix, opt.suffix)
    opt.ign_suffix = ifn(opt.ign_suffix, opt.suffix)

    # Define Path
    opt.train_id = opt.base_train_id
    if opt.baseline:
        saved_dn = Path(f"saved/baseline/{opt.train_id}")
    else:
        saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    summary_pipline = SummaryPipline(
        sample_smi_fn = saved_dn / f"sample/smi{opt.suffix}.json",
        sample_ref_smi_fn = saved_dn / f"test_smi{opt.test_smi_suffix}.json",
        docking_score_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_score{opt.docking_suffix}.json",
        docking_lookup_fn = saved_dn / f"evalute{opt.evaluate_suffix}/batch_docking_lookup{opt.docking_suffix}.json",
        qedsa_fn = saved_dn / f"evalute{opt.evaluate_suffix}/qedsa{opt.suffix}.json",
        dist_fn = saved_dn / f"evalute{opt.evaluate_suffix}/dist{opt.suffix}.json",
        retro_db = Path(opt.retro_db) if "retro_db" in opt else None,
        ign_score_fn = saved_dn / f"evalute{opt.evaluate_suffix}/ign_score{opt.ign_suffix}.json",
    )

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/summary{opt.suffix}.log")]
    final_fn = use_path(file_path=saved_dn / f"evalute{opt.evaluate_suffix}/summary{opt.suffix}.json")

    # Initialize log
    init_logging(log_fn)
    logging.info(f'opt: {opt}')

    # Compute
    toc = Toc()
    summary_pipline.compute()

    # Save
    final_dic = dict()
    for k, v in summary_pipline.final_dic.items():
        if any((
            'med' in k,
            # 'std' in k,
            # 'ref' in k,
        )):
            continue
        _k = '|'.join(k)
        if "scale100" in k:
            _v = round(v, 2)
        elif isinstance(v, float):
            _v = round(v, 3)
        else:
            _v = v
        final_dic[_k] = _v

    logging.info(final_dic)
    auto_dump(final_dic, final_fn, json_indent=True)
    logging.info(f"Saved -> final_fn: {final_fn}, toc: {toc():.3f}")

    if opt.export_fn is not None:
        export_fn = use_path(file_path=opt.export_fn)
        auto_dump(final_dic, export_fn, json_indent=True)
        logging.info(f"Saved -> export_fn: {export_fn}")

    logging.info("Done!")
