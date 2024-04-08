from functools import lru_cache
from pathlib import Path
from tpot import TPOTRegressor
import numpy as np
import re
import joblib
from typing import Iterable, List, Dict, Tuple, Union
from sklearn.pipeline import Pipeline
from halo import Halo
from rdkit import Chem

from ..utils import Mol, StrPath, auto_load, auto_dump, use_path, smi_to_fp, pretty_kv

DEFAULT_SAVED_DN = "saved/regress_tox/20230402_135859_c91a"

class APIRegressTox:

    tpot_scales = ["tiny", "normal", "full"]

    def __init__(self,
                 saved_dn: StrPath=DEFAULT_SAVED_DN,
                 training: bool=False,
                 dimensionless: bool=True) -> None:
        saved_dn = use_path(dir_path=saved_dn)
        self.fn_dic = {
            "tpot_pipeline": saved_dn / "tpot_pipeline.py",
            "fitted_pipeline": saved_dn / "fitted_pipeline.joblib",
            "evaluated_individuals": saved_dn / "evaluated_individuals.json"
        }
        self.training = training
        if not training:
            self.init_inference_from_saved_dn(saved_dn)
        else:
            self.fitted_pipeline: Union[Pipeline, None] = None
        self.dimensionless = dimensionless

        self.chem_predict_cached = lru_cache(maxsize=1<<16)(self.chem_predict)

    def _dataset_to_array(self, d: List[Dict]):
        x_arr = np.array([x["fp"] for x in d], dtype=np.float64)
        y_arr = np.array([x["val"] for x in d], dtype=np.float64)
        return x_arr, y_arr

    def train(self, train_set: List[Dict], test_set: List[Dict], 
              scale: str="full", n_jobs: int=1) -> Tuple[str, float]:
        """Use TPOT to automate the selection of the best machine learning pipeline.
        """
        if not self.training:
            raise Exception(f"Please switch to training status, otherwise the "
                            f"file will be overwritten")
        
        if scale == "tiny":
            tpot = TPOTRegressor(generations=1, population_size=10, 
                                verbosity=2, random_state=20230402, n_jobs=n_jobs,
                                config_dict='TPOT light')
        elif scale == "normal":
            tpot = TPOTRegressor(generations=5, population_size=50, 
                                verbosity=2, random_state=20230402, n_jobs=n_jobs)
        elif scale == "full":
            tpot = TPOTRegressor(generations=100, population_size=100, 
                                verbosity=2, random_state=20230402, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknow scale: {scale}")
        
        x_arr, y_arr = self._dataset_to_array(train_set)
        tpot.fit(x_arr, y_arr)
        self.fitted_pipeline: Pipeline = tpot.fitted_pipeline_

        tpot.export(self.fn_dic["tpot_pipeline"])
        joblib.dump(tpot.fitted_pipeline_, self.fn_dic["fitted_pipeline"])
        auto_dump(tpot.evaluated_individuals_, self.fn_dic["evaluated_individuals"], json_indent=True)

        x_arr, y_arr = self._dataset_to_array(test_set)
        test_score: float = tpot.score(x_arr, y_arr)

        return test_score

    def chem_predict(self, smi: Union[str, Iterable[str]]) -> Union[float, List[float]]:
        """Predicting toxicity values from SMILES string(s)
        """
        assert self.fitted_pipeline is not None, "Looks like I haven't trained yet"

        isnot_batch = isinstance(smi, str)
        smis = [smi] if isnot_batch else list(smi)
        fps = [smi_to_fp.maccs(x) for x in smis]
        x_arr = np.array(fps, dtype=np.float64)
        y_arr = self.fitted_pipeline.predict(x_arr)
        val = y_arr[0] if isnot_batch else y_arr.tolist()

        return val

    def tox_score_for_mso(self, mol: Mol) -> float:
        """For dimensionless means the smaller the better
        """
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            val = self.chem_predict_cached(smi)
        else:
            val = None

        if self.dimensionless:
            # [0.013762223, 9.647902639] -> 10
            score = val if val is not None else 10
        else:
            # 0 <- [0.000249993, 463962.216]
            score = val if val is not None else 0
        return score

    @property
    def tox_desirability_for_mso(self):
        """
        For dimensionless means the smaller the better, 
        output should be the larger the better.
        """
        if self.dimensionless:
            # [0.013762223, 9.647902639]
            return [{"x": 0.0, "y": 1.0}, {"x": 5.0, "y": 0.1}, {"x": 10.0, "y": 0.0}]
            # return [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": -10.0}]
        else:
            # [0.000249993, 463962.216]
            return [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.1}, {"x": 1000.0, "y": 1.0}]

    def init_inference_from_saved_dn(self, saved_dn: Path) -> None:
        """Recover api from saved_dn
        """
        desc = f'Initialize {self.__class__.__name__} from {saved_dn.name}'

        spinner = Halo(text=desc)
        spinner.start()

        # Pipeline
        spinner.text = desc + f" -> fitted_pipeline: {self.fn_dic['fitted_pipeline']}"
        if not self.fn_dic["fitted_pipeline"].exists():
            raise ValueError("Looks like I haven't trained yet")
        self.fitted_pipeline: Pipeline = joblib.load(self.fn_dic["fitted_pipeline"])
        
        # Info
        opt_fn = use_path(file_path=saved_dn / "opt.json", new=False)
        spinner.text = desc + f" -> opt_fn: {opt_fn}"
        opt = auto_load(opt_fn, to_edict=True)

        spinner.text = desc + f" -> tpot_pipeline: {self.fn_dic['tpot_pipeline']}"
        with open(self.fn_dic["tpot_pipeline"], 'r') as f:
            tpot_pipeline = f.read()
        m = re.search(r"# Average CV score on the training set was: (?P<score>[+-]?((\d+\.?\d*)|(\.\d+)))", 
                      tpot_pipeline)
        if m is not None:
            cv_score = float(m["score"])
            spinner.text = desc + f" -> evaluated_individuals: {self.fn_dic['evaluated_individuals']}"
            evaluated_individuals = auto_load(self.fn_dic["evaluated_individuals"])
            k, v = min(evaluated_individuals.items(), key=lambda x: abs(x[1]["internal_cv_score"] - cv_score) \
                       if x[1]["internal_cv_score"] is not None else np.inf)
            if abs(v["internal_cv_score"] - cv_score) < 1e-10:
                pipeline_desc = k
            else:
                pipeline_desc = None
        else:
            cv_score = None

        info = dict(tpot_scale=opt.tpot_scale, cv_score=cv_score, pipepline_desc=pipeline_desc)
        spinner.succeed(desc + f" -> Done! {pretty_kv(info, ndigits=5)}")
    