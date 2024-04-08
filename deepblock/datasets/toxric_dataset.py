from pathlib import Path
from typing import Dict, List, Union, Tuple
import logging
import warnings
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from ..utils import StrPath, use_path, download_pbar, convert_bytes, smi_to_fp, split_pro_to_idx

class ToxricDataset:
    TOXRIC_DOWNLOAD_URL = {
        ("Acute Toxicity", "mouse_intraperitoneal_LD50",): "https://toxric.bioinforai.tech/jk/DownloadController/DownloadToxicityInfo?toxicityId=44",
        ("Acute Toxicity", "mouse_intraperitoneal_LD50", "dimensionless",): "https://toxric.bioinforai.tech/jk/DownloadController/DownloadAcuteToxicity2?toxicityId=44",
        ("Feature Space", "MACCS",): "https://toxric.bioinforai.tech/jk/DownloadController/DownloadFeatureInfo?featureId=21",
    }
    def __init__(self, cache_dn: StrPath, 
                 dimensionless: bool=True) -> None:
        _cache_dn = Path(cache_dn)
        self.fn_dic: Dict[Union[Tuple[str], str], Path] = {}
        self.mem: Dict[Union[Tuple[str, ...], str], object] = {}
        for k in self.TOXRIC_DOWNLOAD_URL:
            _fn = ','.join(x.lower().replace(' ', '_') for x in k) + '.csv'
            self.fn_dic[k] = _cache_dn / "download" / _fn
        self.dimensionless = dimensionless

    def download(self) -> 'ToxricDataset':
        if self.dimensionless:
            self._download(("Acute Toxicity", "mouse_intraperitoneal_LD50", "dimensionless",))
        else:
            self._download(("Acute Toxicity", "mouse_intraperitoneal_LD50",))
        self._download(("Feature Space", "MACCS",))
        return self

    def _download(self, k: Union[Tuple[str], str]) -> None:
        _url = self.TOXRIC_DOWNLOAD_URL[k]
        _fn = use_path(file_path=self.fn_dic[k])
        if not _fn.exists():
            print(f"Download -> {_url} to {_fn}")
            try:
                _bytes = download_pbar(_url, desc=f"↓ {_fn.name}")
            except Exception as err:
                raise Exception(f"{_fn.name} {_url} download failed due to {err}! ")
            with open(_fn, 'wb') as f:
                f.write(_bytes)
            print(f"Done -> size: {convert_bytes(len(_bytes))}")

    def preprocess(self, check_fp=False) -> 'ToxricDataset':
        if self.dimensionless:
            _fn = self.fn_dic[("Acute Toxicity", "mouse_intraperitoneal_LD50", "dimensionless",)]
            tox_df = pd.read_csv(_fn)
            tox_df = tox_df[["TAID", "SMILES", "mouse_intraperitoneal_LD50"]].rename(
                columns={"mouse_intraperitoneal_LD50": "VALUE"})
        else:
            tox_df = self.fn_dic[("Acute Toxicity", "mouse_intraperitoneal_LD50",)]
            tox_df = tox_df[["TAID", "SMILES", "Toxicity Value"]].rename(
                columns={"Toxicity Value": "VALUE"})
            
        tox_lst: List[Dict] = tox_df\
            .sort_values('TAID', key=lambda x: x.str.split('-').str.get(1).astype(int))\
            .apply(lambda row: {'taid': row["TAID"], 'smi': row["SMILES"], 'val': row["VALUE"]}, axis=1)\
            .tolist()

        _fn = self.fn_dic[("Feature Space", "MACCS",)]
        fp_df = pd.read_csv(_fn)
        fp_df = fp_df.set_index('TAID')[[f"MA_{n}" for n in range(1, 168)]]
        fp_not_found_taid_lst = []
        for x in tox_lst:
            try:
                x["fp"] = tuple(fp_df.loc[x["taid"]].to_list())
            except KeyError:
                x["fp"] = smi_to_fp.maccs(x["smi"])
                fp_not_found_taid_lst.append(x["taid"])

        if len(fp_not_found_taid_lst) > 0:
            warnings.warn(f"TAID: {fp_not_found_taid_lst} fingerprints are not readily available, "
                          f"replace them with calculated results.")

        if check_fp:
            for x in tqdm(tox_lst, desc="Check MACCS fingerprint"):
                cal_fp = smi_to_fp.maccs(x["smi"])
                assert cal_fp == x["fp"], f"Fingerprint conflict: {x['taid']} {x['smi']}\n{x['fp']}\n{cal_fp}"

        self.mem["tox"] = tox_lst
        return self

    def split(self, split_pro: Dict[str, float]=None, 
              ret_tup: Tuple[str, ...]=None) -> Union[List[Dict], Dict[str, List], Tuple[List[Dict]]]:
        if "tox" not in self.mem:
            raise Exception("Run preprocess first!")
        tox_lst = self.mem["tox"]
        if split_pro is None:
            return tox_lst
        else:
            split_idx_dic = split_pro_to_idx(split_pro, total=len(tox_lst), seed=20230402)
            split_tox_dic = {k: [tox_lst[i] for i in v] for k, v in split_idx_dic.items()}
            if ret_tup is None:
                return split_tox_dic
            else:
                return tuple(split_tox_dic[k] for k in ret_tup)
            