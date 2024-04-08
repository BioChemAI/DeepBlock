import argparse
from itertools import groupby
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
STAGE = "evaluate_summary.optimize"


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate: Summary for optimize - CVAE Complex')

    parser.add_argument("--base-train-id", type=str, required=True)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--docking-suffix", type=str)
    opt = mix_config(parser, None)
    return opt

class SummaryPipline:

    def __init__(self, **fns) -> None:
        self.fns = edict(fns)
        self.pipline = {
            "Count": self.item_count,
            "Vina Affinity": self.item_vina,
            "Distribution": self.item_dist,
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
        if not isinstance(arr, np.ndarray):
            arr = np.fromiter(arr, float)
        for mms, value in zip(
                ('mean', 'med', 'std'), 
                (np.mean(arr), np.median(arr), np.std(arr))):
            dic[(*names, mms)] = float(value)

    @staticmethod
    def groupby_cid(oid_dic: Dict):
        lst = [{**v, "cid": k.split(';')[0], "oid": k} for k, v in oid_dic.items()]
        cid_dic = {k: list(filter(lambda x: x["cid"] == k, lst)) for k in set(map(lambda x: x["cid"], lst))}
        return cid_dic

    # 0. Count
    def item_count(self, name):
        try:
            prd_res_dic = auto_load(self.fns.prd_res_fn)
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            self.final_dic[(name, 'prd', 'sum')] = len(prd_res_dic)
            self.final_dic[(name, 'ref', 'sum')] = len(prd_res_dic)

    # 1. Vina Affinity
    def item_vina(self, name):
        try:
            prd_res_dic = auto_load(self.fns.prd_res_fn)
            docking_score_lst = auto_load(self.fns.docking_score_fn)             # [{id(hash), score}, ...]      Data will be missing!!
            docking_lookup_dic = edict(auto_load(self.fns.docking_lookup_fn))    # {ref: {id: hash, ...}, prd: {id: {smi: hash, ...}, ...}}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            hash_to_score = {x['id']: x['affinity'] for x in docking_score_lst}
            cid_to = edict()
            optim_dic = self.groupby_cid(prd_res_dic)

            for mod in ("ref", "prd"):
                cid_to[f"{mod}_mean"]: Dict[str, float] = dict()
                cid_to[f"{mod}_miss"]: Dict[str, int] = dict()
            cid_to.high: Dict[str, float] = dict()

            comp_arr_dic = dict()
            for cid, lst in optim_dic.items():
                score_lst_dic = dict()
                for mod in ("ref", "prd"):
                    h_lst = [docking_lookup_dic[x["oid"]][mod] for x in lst]
                    score_lst_dic[mod] = [hash_to_score.get(h, None) for h in h_lst]
                    score_lst = [x for x in score_lst_dic[mod] if x is not None]
                    cid_to[f"{mod}_mean"][cid] = np.mean(score_lst)
                    cid_to[f"{mod}_miss"][cid] = len(h_lst) - len(score_lst)
                # Compare
                comp_arr = np.array([(r, p) for r, p in zip(score_lst_dic["ref"], score_lst_dic["prd"]) if (r is not None) and (p is not None)])
                comp_arr_dic[cid] = comp_arr
                cid_to.high[cid] = np.mean(comp_arr[:,1] <= comp_arr[:,0]) * 100
            comp_arr_all = np.concatenate(list(comp_arr_dic.values()))

            is_between = lambda x, a, b: np.logical_and(x>a, x<=b)
            def comp_between(arr, a=-np.inf, b=np.inf):
                arr = arr[is_between(arr[:,0], a, b)]
                return len(arr), np.mean(arr[:,0]), np.mean(arr[:,1]), np.mean(arr[:,1] <= arr[:,0])
            comp_between(comp_arr_all, a=-7, b=np.inf)
            for a in np.arange(-15, -5, 0.1):
                res = comp_between(comp_arr_all, a=a, b=np.inf)
                res = [str(round(x, 4) if isinstance(x, float) else x) for x in res]
                self.final_dic[(name, 'comp', str(round(a, 4)))] = '|'.join(f"{x:<8}" for x in res)
    
            def comp_percentile(comp_arr_dic, p):
                dic = {}
                for k, v in comp_arr_dic.items():
                    t = np.percentile(v[:,0], 100-p)
                    dic[k] = (t,) + comp_between(v, t)
                return dic
            for p in np.arange(5, 100, 5):
                dic = comp_percentile(comp_arr_dic, p)
                res = np.mean(list(dic.values()), axis=0)
                res = [str(round(x, 4) if isinstance(x, float) else x) for x in res]
                self.final_dic[(name, 'comp_per', str(round(p, 4)))] = '|'.join(f"{x:<8}" for x in res)

            # comp_percentile(comp_arr_dic, 25)
            # {k: v for k, v in comp_percentile(comp_arr_dic, 25).items() if v[3] < -7 and v[4] > 0.7 and v[0] > -7}

            for mod in ("ref", "prd"):
                mean_arr = np.array(list(cid_to[f"{mod}_mean"].values()))
                self.append_mms_to_dic(self.final_dic, mean_arr, name, mod)
                self.final_dic[(name, 'miss', mod, 'sum')] = sum(cid_to[f"{mod}_miss"].values())
            mean_arr = np.array(list(cid_to.high.values()))
            self.append_mms_to_dic(self.final_dic, mean_arr, name, 'high', 'scale100')

    # 3. Distribution
    def item_dist(self, name):
        try:
            dist_dic = auto_load(self.fns.dist_fn)                               # {id: smi, ...}
        except Exception as err:
            logging.warning(f'Item {name} drop, because {repr(err)}')
        else:
            ktbn = {
                "similarity": ("Diversity",),
                "scaffold_similarity": ("Scaffold Diversity",),
            }
            cid_to = edict()
            dist_dic = self.groupby_cid(dist_dic)
            for kt, bn in ktbn.items():
                cid_to[kt] = dict()
                cid_to[kt+"%"] = dict()
                for cid, lst in dist_dic.items():
                    cid_to[kt][cid] = np.mean([x[kt] for x in lst])
                    cid_to[kt+"%"][cid] = np.mean([x[kt] >= 0.4 for x in lst])
                self.append_mms_to_dic(self.final_dic, cid_to[kt].values(), name, *bn)
                self.append_mms_to_dic(self.final_dic, cid_to[kt+"%"].values(), name, *bn, 'scale100')

    # TODO 统计一个ref分数到相似度的分布

if __name__ == '__main__':
    opt = parse_opt()
    opt.docking_suffix = ifn(opt.docking_suffix, opt.suffix)

    # Define Path
    opt.train_id = opt.base_train_id
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")

    ## Input
    summary_pipline = SummaryPipline(
        prd_res_fn = saved_dn / f"optimize/prd_res{opt.suffix}.json",
        docking_score_fn = saved_dn / f"optimize/batch_docking_score{opt.docking_suffix}.json",
        docking_lookup_fn = saved_dn / f"optimize/batch_docking_lookup{opt.docking_suffix}.json",
        dist_fn = saved_dn / f"optimize/dist{opt.suffix}.json",
    )

    ## Output
    log_fn = [use_path(file_path=saved_dn / f"{STAGE}.log"),
              use_path(file_path=saved_dn / f"optimize/summary{opt.suffix}.log")]
    final_fn = use_path(file_path=saved_dn / f"optimize/summary{opt.suffix}.json")

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
            'std' in k,
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

    logging.info("Done!")
