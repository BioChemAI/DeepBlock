import argparse
from collections import Counter, UserList, defaultdict
from dataclasses import asdict, dataclass, fields
import datetime
from functools import reduce, wraps
import gc
from hashlib import sha256
import hashlib
from numbers import Number
import secrets
import shutil
import stat
import sys
import tarfile
import traceback
import warnings
from easydict import EasyDict as edict
from itertools import accumulate, chain
import joblib
import ormsgpack
import pexpect
import requests
import torch
from torch import Tensor
import json
import pickle
import random
import time
from typing import Any, BinaryIO, Callable, Dict, Hashable, Iterable, OrderedDict, Sequence, Tuple, TypeVar, Union, List
import logging
from joblib import Memory, delayed
from pathlib import Path
from contextlib import contextmanager, redirect_stderr
import numpy as np
from rdkit import rdBase, Chem
from io import StringIO
import orjson
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

import yaml
import psutil
from openbabel import openbabel

from ..exceptions import RDKitException

Mol = TypeVar('Mol', bound=Chem.rdchem.Mol)
StrPath = Union[str, Path]

def use_path(*, dir_path: StrPath = None, file_path: StrPath = None, new: bool = True) -> Path:

    assert sum((dir_path is None, file_path is None)) == 1, \
        Exception(f"There is only one in dir_path ({dir_path}) and file_path ({file_path})")

    if dir_path is not None:
        _dir_path = Path(dir_path)
        if new:
            _dir_path.mkdir(exist_ok=True, parents=True)
        else:
            assert _dir_path.exists(), Exception(f"{_dir_path} does not exist")
        return _dir_path

    if file_path is not None:
        _file_path = Path(file_path)
        if new:
            _file_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            assert _file_path.exists(), Exception(f"{_file_path} does not exist")
        return _file_path

def use_memory(cachedir='./cache'):
    use_path(dir_path=cachedir)
    memory = Memory(cachedir, verbose=0)
    return memory

auto_dump_load_suffixes = ['.json', '.pkl', '.yaml', '.msgpack']

def auto_dump(obj: Any, fn: StrPath, use_orjson=True, json_indent=False):
    _path = use_path(file_path=fn)
    if _path.suffix == '.json':
        if use_orjson:
            with open(_path, 'wb') as f:
                f.write(orjson.dumps(obj, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2) \
                    if json_indent else orjson.dumps(obj, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
        else:
            with open(_path, 'w') as f:
                json.dump(obj, f, indent=2) \
                    if json_indent else json.dump(obj, f)
    elif _path.suffix == '.pkl':
        with open(_path, 'wb') as f:
            pickle.dump(obj, f)
    elif _path.suffix == '.yaml':
        with open(_path, 'w') as f:
            yaml.dump(obj, f)
    elif _path.suffix == '.msgpack':
        with open(_path, 'wb') as f:
            f.write(ormsgpack.packb(obj, option=ormsgpack.OPT_NAIVE_UTC | ormsgpack.OPT_SERIALIZE_NUMPY))
    else:
        raise Exception(f"Unknow suffix: {_path.suffix}")

def auto_load(fn: StrPath, use_orjson=True, to_edict=False):
    _path = use_path(file_path=fn, new=False)
    if _path.suffix == '.json':
        if use_orjson:
            with open(_path, 'rb') as f:
                obj = orjson.loads(f.read())
        else:
            with open(_path, 'r') as f:
                obj = json.load(f)
    elif _path.suffix == '.pkl':
        with open(_path, 'rb') as f:
            obj = pickle.load(f)
    elif _path.suffix == '.yaml':
        with open(_path, 'r') as f:
            obj = yaml.safe_load(f)
    elif _path.suffix == '.msgpack':
        with open(_path, 'rb') as f:
            obj = ormsgpack.unpackb(f.read())
    else:
        raise Exception(f"Unknow suffix: {_path.suffix}")
    if to_edict:
        obj = edict(obj)
    return obj

def auto_loadm(fns: Iterable[StrPath], *args, **kwargs):
    return {k: v for fn in fns for k, v in auto_load(fn, *args, **kwargs).items()}

@contextmanager
def rdkit_log_handle():
    """Capture the log of rdkit and hide the output."""
    # https://github.com/rdkit/rdkit/discussions/5435#discussioncomment-3187358
    # Redirect RDKit logs to stderr
    rdBase.LogToPythonStderr()
    with StringIO() as buf:
        with redirect_stderr(buf):
            def rdLog():
                log = buf.getvalue().rstrip()
                return log
            yield rdLog

@contextmanager
def ob_log_handle(output_level: int=0):
    """Disable all but critical messages."""
    # https://stackoverflow.com/questions/50419371/how-to-disable-logged-warnings-in-pybel
    # http://openbabel.org/dev-api/classOpenBabel_1_1OBMessageHandler.shtml
    old_output_level = openbabel.obErrorLog.GetOutputLevel()
    try:
        openbabel.obErrorLog.SetOutputLevel(output_level)
        yield
    finally:
        openbabel.obErrorLog.SetOutputLevel(old_output_level)

def init_logging(log_fn: Union[StrPath, Tuple[StrPath], List[StrPath]] = None):
    # https://docs.python.org/zh-cn/3/library/logging.html#logrecord-attributes
    log_fns = log_fn if isinstance(log_fn, (tuple, list)) else (log_fn,) if log_fn is not None else tuple()
    log_fns = [use_path(file_path=x) for x in log_fns]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] | %(message)s",
        handlers=[
            *(logging.FileHandler(x) for x in log_fns),
            logging.StreamHandler()
        ]
    )

def unique_by_key(lst: Iterable[Dict], *keys: Hashable) -> List:
    if not keys:
        keys = lst[0].keys()
    return list({
        tuple(
            tuple(_obj[k]) if isinstance(_obj[k], list) else _obj[k] 
            for k in keys
        ): _obj 
        for _obj in lst
    }.values())

class Toc():
    """Simple reusable timer
    """
    def __init__(self) -> None:
        self.t = time.time()
    def __call__(self) -> float:
        _t = self.t
        self.t = time.time()
        return self.t - _t

def split_pro_to_idx(split_pro_dic: Dict[str, float], total: int, seed: Any):
    """Example
    ```
    >>> split_pro_to_idx({"train": 0.8, "valid": 0.1, "test": 0.1}, 15, 47)
    {'train': [5, 1, 6, 8, 7, 9, 14, 4, 10, 3, 11, 0], 'valid': [13, 12], 'test': [2]}
    ```
    """
    q = random.Random(seed).sample(range(total), k=total)
    assert sum(split_pro_dic.values()) == 1
    r = tuple(accumulate(round(p * total) for p in split_pro_dic.values()))
    split_idx_dic = {k: q[a:b] for k, a, b in zip(split_pro_dic.keys(), (0, *r[:-1]), (*r[:-1], total))}
    assert tuple(chain.from_iterable(split_idx_dic.values())) == tuple(q)
    return split_idx_dic

def mix_config(parser: argparse.ArgumentParser, script_fn: StrPath=None) -> edict:
    """Expand config file for argument parser (Load default config file first: `default_config.yaml`.
    Then `dev_config.yaml` if in development mode.)
    """
    parser.add_argument("--dev", action="store_true", 
                        help="Development mode (limited data set).")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="Disable Wandb.")
    parser.add_argument("--config", nargs='+', type=str, default=[], 
                        help="Additional configuration files, overwriting forward.")
    args = parser.parse_args()
    _config_fns: List = args.config.copy()
    _preset_fns = []

    _preset_names = ['default_config']
    _preset_names.append('dev_config' if args.dev else 'prod_config')

    if script_fn:
        for name in _preset_names:
            for suffix in auto_dump_load_suffixes:
                _fn: Path = Path(script_fn).with_name(name + suffix)
                if _fn.exists():
                    try:
                        _fn = _fn.relative_to(Path.cwd())
                    except ValueError:
                        pass
                    _preset_fns.append(str(_fn.as_posix()))
    opt = edict({**auto_loadm(_preset_fns + _config_fns), **vars(args)})
    return opt

def pretty_kv(dic: Dict[str, Any], ndigits: int=2, 
            prefix: str='', sep: str=', ', colon: str=': ') -> str:
    _dic: Dict[str, str] = dict()
    for k, v in dic.items():
        _dic[str(k)] = str(round(v, ndigits) if isinstance(v, Number) else v)
    return prefix + sep.join(f"{k}{colon}{v}" for k, v in _dic.items() if not k.startswith('_'))

def summary_arr(arr: List[Any], key: Callable=None, 
                return_str=True, remove_none=True) -> Union[Dict[str, Number], str]:
    dic = dict(len=len(arr))
    if key:
        arr = list(map(key, arr))
    if remove_none:
        arr = [x for x in arr if x is not None]
        dic = dict(**dic, none=dic['len'] - len(arr))
    if len(arr) > 0:
        arr = np.array(arr)
        dic = dict(**dic, 
            mean=np.mean(arr), var=np.var(arr), min=np.min(arr), max=np.max(arr), 
            **{f'{q}%': np.percentile(arr, q) for q in (1, 25, 50, 75, 99)})
    if return_str:
        return pretty_kv(dic)
    else:
        return dic

def init_random(random_seed):
    """Initialize random seed of random, numpy and torch.
    """
    r = random.Random(random_seed)
    random.seed(int(r.random() * (1<<32)))
    np.random.seed(int(r.random() * (1<<32)))
    torch.manual_seed(int(r.random() * (1<<32)))

def sorted_seqs(*seqs: Sequence, return_idx=False):
    """Stable sorting of multiple sequences, first by lengths, then by items.
    """
    idx, *res = zip(*sorted(enumerate(zip(*seqs)), key=lambda x: (*map(len, x[1]), *x[1], x[0])))
    if return_idx:
        return idx
    else:
        return res

def pick_by_idx(*lsts: List, idx_lst: List[int], inplace=False):
    _lsts = []
    for lst in lsts:
        _lst = [lst[i] for i in idx_lst]
        if inplace:
            lst[:] = _lst
        _lsts.append(_lst)
    return _lsts

def get_time_str():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def generate_train_id(is_dev=False, suffix=None):
    time_str = get_time_str()
    token_hex = secrets.token_hex(2)
    train_id = f'{time_str}_{token_hex}'
    if is_dev:
        train_id = f'[dev]{train_id}'
    if suffix:
        train_id = f'{train_id}_{suffix}'
    return train_id

@dataclass
class VAELossItem():
    recons_loss: float = 0
    kld_loss: float = 0
    loss: float = 0
    total: int = 0 # No reduction
    _sca: float = 1

    def __add__(self, other):
        assert isinstance(other, VAELossItem)
        dic_a = asdict(self)
        dic_b = asdict(other / (self._sca / other._sca))
        dic_c = {k: dic_b[k] + v for k, v in dic_a.items() if k != '_sca'}
        return self.__class__(**dic_c, _sca=self._sca)
    
    def __truediv__(self, other):
        assert isinstance(other, Number)
        dic_a = asdict(self)
        dic_c = {k: v / other for k, v in dic_a.items() if k.endswith('loss')}
        return self.__class__(**dic_c, total=self.total, _sca=other)

    def __mul__(self, other):
        assert isinstance(other, Number)
        dic_a = asdict(self)
        dic_c = {k: v * other for k, v in dic_a.items() if k.endswith('loss')}
        return self.__class__(**dic_c, total=self.total, _sca=1/other)

    def mean(self):
        return self / self.total if self.total > 0 else self.__class__()

@dataclass
class CVAELossItem(VAELossItem):
    bow_loss: float = 0
    rel_loss: float = 0

class VAELossList(UserList):
    def sum(self) -> VAELossItem:
        return reduce(lambda x, y: x + y, self.data)
    def mean(self) -> VAELossItem:
        return self.sum().mean()

class CVAELossList(UserList):
    def sum(self) -> CVAELossItem:
        return reduce(lambda x, y: x + y, self.data)
    def mean(self) -> CVAELossItem:
        return self.sum().mean()

class BetaVAEScheduler():
    allow_mode = ('fix', 'annealing')
    def __init__(self, mode: str='fix', value: Union[Number, Dict]=1, last_epoch=-1):
        assert mode in self.allow_mode, f"Mode {self.mode} not allowed"
        self.mode = mode
        self.value = value
        self.last_epoch = last_epoch
        self.step()

    def step(self):
        self.last_epoch += 1
        self.last_beta = self.get_beta()

    def get_beta(self):
        v = self.value
        if self.mode == 'fix':
            beta = v
        elif self.mode == 'annealing':
            t = self.last_epoch % v.cycle if v.cycle > 0 else self.last_epoch
            if t <= v.min_point:
                beta = v.min_beta
            elif t < v.max_point:
                beta = (t - v.min_point) / (v.max_point - v.min_point) * \
                    (v.max_beta - v.min_beta) + v.min_beta
            else:
                beta = v.max_beta
        else:
            raise f"Mode {self.mode} not allowed"
        return beta

    def get_last_beta(self):
        return self.last_beta

class CheckpointManger():

    state_keys = ['last_epoch', 'save_step', 'values', 'epochs']

    def __init__(self, weights_dn: Path, best_fn: Path, 
                save_step: int=-1, last_epoch: int=-1, auto_clean: bool=True) -> None:
        self.weights_dn = weights_dn
        self.best_fn = best_fn
        self.save_step = save_step
        self.last_epoch = last_epoch
        self.values = dict()
        self.epochs = dict()
        self.auto_clean = auto_clean

    def epoch_to_weight_fn(self, epoch: int) -> Path:
        assert epoch >= 0 and isinstance(epoch, int)
        return self.weights_dn / f"{epoch:05d}.pt"

    def submit(self, epoch: int, values: Dict, weight: OrderedDict[str, Tensor]):
        """Submit weight and values.

        Parameters:

        epoch: Start with 0
        """

        weight_fn = self.epoch_to_weight_fn(epoch)
        torch.save(weight, weight_fn)

        dec_dic = {}
        for k, v in values.items():
            if k not in self.values:
                dec_dic[k] = f'new->{v:.5f}'
                self.values[k] = v
                self.epochs[k] = epoch
            elif self.values[k] > v:
                dec_dic[k] = f'{self.values[k]:.5f}->{v:.5f}'
                self.values[k] = v
                self.epochs[k] = epoch       
        self.last_epoch = epoch
        self.dump()

        if self.auto_clean:
            self.clean()
        return dec_dic, weight_fn

    def clean(self):
        white_lst = tuple(map(self.epoch_to_weight_fn, [
            self.last_epoch,
            *self.epochs.values(), 
            *range(-1, self.last_epoch+1, self.save_step)[1:]
        ]))
        for weight_fn in self.weights_dn.glob("*.pt"):
            if weight_fn not in white_lst:
                weight_fn.unlink()

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.state_keys}

    def __setstate__(self, state):
        return {setattr(self, k, state[k]) for k in self.state_keys}

    def load(self):
        self.__setstate__(auto_load(self.best_fn))

    def dump(self):
        auto_dump(self.__getstate__(), self.best_fn, json_indent=True)

    def pick(self, choice: str) -> Path:
        self.load()

        # 1. Latest
        if choice == "latest":
            return self.epoch_to_weight_fn(self.last_epoch)

        # 2. Value
        if choice in self.epochs:
            return self.epoch_to_weight_fn(self.epochs[choice])
        
        # 3. Number ["23", "033", "-1", ...]
        try:
            epoch = int(choice)
            if epoch < 0:
                epoch = self.last_epoch + epoch + 1
            return self.epoch_to_weight_fn(epoch)
        except ValueError:
            pass

        raise Exception(f"Unable to determine weight: {choice}")

def sha256_dic(dic: Dict) -> bytes:
    return sha256(json.dumps(dic).encode()).digest()

def convert_bytes(num: int):
    """Convert bytes to MB, GB, etc.
    """
    for unit in ('Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'):
        if num < 1024:
            break
        else:
            num /= 1024

    return f'{num} {unit}' if unit == 'Bytes' else f'{num:.3f} {unit}'

def get_file_size(fn: StrPath, pretty=False):
    """https://stackoverflow.com/a/39988702/16407115
    """
    size = Path(fn).stat().st_size
    return convert_bytes(size) if pretty else size

def pretty_dataclass_repr(obj):
    _lst = []
    for field in fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, np.ndarray):
            value = f"array([shape({'x'.join(str(x) for x in value.shape)})])"
        elif isinstance(value, torch.Tensor):
            value = f"tensor([shape({'x'.join(str(x) for x in value.shape)})])"
        elif isinstance(value, list):
            value = f"list([len({len(value)})])"
        elif isinstance(value, str):
            value = f"'{value}'"
        _lst.append(f"{field.name}={value}")
    return f"{obj.__class__.__qualname__}({', '.join(_lst)})"

def norm_rel_fn(arr: np.ndarray) -> np.ndarray:
    alpha = 1e-10
    arr = 1 - (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * (1 - alpha) + alpha
    return arr / arr.sum()

@contextmanager
def gc_disabled(mem_threshold: Number=-1):
    """Temporarily disable `gc`.

    Args:
        mem_threshold (Number, optional): Available memory (GB) threshold, 
        must be greater than 0. Defaults to -1.
    """
    if mem_threshold:
        mem_status = psutil.virtual_memory()
        act = mem_status.available > mem_threshold<<30
    else:
        act = True

    if act:
        gc_isenabled = gc.isenabled()
        gc.disable()
        try:
            yield
        finally:
            if gc_isenabled:
                gc.enable()
    else:
        yield

@torch.no_grad()
def assert_prob(x: Tensor, name: str='x', threshold: float=1e-5):
    y = torch.sum(x, dim=-1)
    e = torch.abs(y-1).gt(threshold)
    if e.any():
        raise ValueError(f"The sum of discrete distribution probabilities "
                         f"{name} needs to be 1, but {y[e]}")
    
def ifn(value, default):
    """Function: `return default if value is None else value`.
    """
    return default if value is None else value

def exp_rel_fn(arr: np.ndarray) -> np.ndarray:
    sigma = 20
    gamma = 2.5
    arr = np.exp(-(arr/sigma)**gamma)
    return arr / arr.sum()

TRelFn = Callable[[np.ndarray], np.ndarray]

rel_fn_dic: Dict[str, TRelFn] = {
    "norm": norm_rel_fn,
    "exp": exp_rel_fn
}

def download_pbar(url: str, timeout: Number=60, desc: str='Download', 
                  chunk_size: int=1024, file_handle: BinaryIO=None):
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = response.headers.get('content-length', None)
    if total is not None: total = int(total)
    pbar = tqdm(desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024)
    if file_handle is None:
        data = bytearray()
        for content in response.iter_content(chunk_size=chunk_size):
            pbar.update(len(content))
            data.extend(content)
        pbar.close()
        return data
    else:
        size = 0
        for content in response.iter_content(chunk_size=chunk_size):
            pbar.update(len(content))
            size += len(content)
            file_handle.write(content)
        pbar.close()
        return size
    
class ChildUnexpectedTerminationException(Exception):
    """The exception pexpect child terminated abnormally."""
    def __init__(self, child: 'pexpect.pty_spawn.SpawnBase'):
        # Exception must being able to dump normally
        if isinstance(child, pexpect.pty_spawn.SpawnBase):
            self.short_message = (f'({child.pid}) {" ".join(child.args)}\n'
                       f'exit({child.exitstatus}): {child.signalstatus}')
            message = (f'{self.short_message}\n'
                       f'{child.before.decode() if isinstance(child.before, (bytes, bytearray)) else child.before}')
        else:
            message = str(child)
        super().__init__(message)

def child_safe_wait(child: 'pexpect.pty_spawn.SpawnBase', is_warn=False, is_ignore=False):
    """Wait for the pexpect child to terminate safely"""
    child.expect(pexpect.EOF)
    child.close()
    if child.exitstatus != 0:
        err = ChildUnexpectedTerminationException(child)
        if is_ignore:
            return err
        elif is_warn:
            warnings.warn(f'{repr(err)}')
            return err  
        else:
            raise err

def chmod_plus_x(f: StrPath):
    Path(f).chmod(f.stat().st_mode | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

def sequential_file_hash(fn: StrPath, h: 'hashlib._Hash', block_size: int=128<<10):
    b  = bytearray(block_size)
    mv = memoryview(b)
    with open(fn, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h

def file_hash(fn: StrPath, h: 'hashlib._Hash'):
    with open(fn, 'rb') as f:
        h.update(f.read())
    return h

# Blueprint for Decorators
# https://book.pythontips.com/en/latest/decorators.html
# https://stackoverflow.com/a/42581103/16407115

def ignore_exception(func: Callable):
    """Catch function exceptions and return None."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            arg_str_lst = [*(f'{v}' for v in args), 
                *(f'{k}={v}' for k, v in kwargs.items())]
            logging.error(f"{func.__name__}"
                          f"({', '.join(arg_str_lst)})"
                          f" -> \n{repr(err)}")
            traceback.print_exc()
            result = None
        return result
    return wrapper

def return_exception(func: Callable):
    """Catch and return function exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            arg_str_lst = [*(f'{v}' for v in args), 
                *(f'{k}={v}' for k, v in kwargs.items())]
            logging.error(f"{func.__name__}"
                          f"({', '.join(arg_str_lst)})"
                          f" -> \n{repr(err)}")
            traceback.print_exc()
            result = err
        return result
    return wrapper

def delayed_return_exception(func: Callable):
    """Catch and return function exceptions and 
    capture the arguments of a function."""
    return delayed(return_exception(func))

def vin(value, collection: Sequence):
    """Test whether the value is in the collection, 
    otherwise throw an exception.
    """
    if value not in collection:
        raise ValueError(f'Value {value} not in {collection}')

@contextmanager
def hook_joblib(cb: Callable[[int, int, float], None]):
    """Context manager to patch joblib BatchCompletion
    cb(batch_size, n_completed_tasks, elapsed_time, result_lst)
    Ref: https://stackoverflow.com/a/58936697/16407115
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            cb(self.batch_size, 
               self.parallel.n_completed_tasks+1,
               time.time() - self.parallel._start_time,
               self.parallel._output)
            return super().__call__(*args, **kwargs)

    OldBatchCompletionCallBack = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = OldBatchCompletionCallBack

class LoggingTrackerJoblibCallback:
    def __init__(self, total=None, blocks=100, interval=None) -> None:
        self.total = total
        if total is None:
            if interval is not None:
                self.interval = interval
            else:
                raise ValueError("Give me interval if total is None")
        else:
            if blocks is not None:
                self.interval = max(1, total // blocks)
            elif interval is not None:
                self.interval = interval
            else:
                raise ValueError("Give me blocks or interval")
        self.delayed_tasks = 0
        self.delayed_err = 0
        self.next_report_tasks = 0
        logging.info(f"Tracking -> {self.total} tasks total, report every {self.interval} tasks")

    def __call__(self, batch_size, n_completed_tasks, elapsed_time, result_lst) -> None:
        result_lst = ifn(result_lst, [])
        if n_completed_tasks >= self.next_report_tasks:
            info_lst = []
            if self.total is None:
                info_lst.append(f"progress: {n_completed_tasks}/?")
            else:
                info_lst.append(f"progress: {n_completed_tasks}/{self.total}={round(n_completed_tasks/self.total*100 if self.total > 0 else 0, 3)}%")

            delayed_tasks = len(result_lst)
            if delayed_tasks != self.delayed_tasks:
                delayed_err = sum(isinstance(result, Exception) for result in result_lst)
                self.delayed_tasks = delayed_tasks
                self.delayed_err = delayed_err

            info_lst.append(f"error: {self.delayed_err}/{self.delayed_tasks}="
                            f"{round(self.delayed_err/self.delayed_tasks*100 if self.delayed_tasks > 0 else 0, 3)}%")

            info_lst.append(f"elapsed_time: {datetime.timedelta(seconds=elapsed_time)}")
            if self.total is not None:
                remain_time = elapsed_time * (self.total / n_completed_tasks - 1)
                info_lst.append(f"remain_time: {datetime.timedelta(seconds=remain_time)}")

            logging.info(f"Tracking -> {', '.join(info_lst)}")
            if self.total is None:
                self.next_report_tasks = (n_completed_tasks // self.interval + 1) * self.interval
            else:
                self.next_report_tasks = min(self.total, (n_completed_tasks // self.interval + 1) * self.interval)

class TqdmTrackerJoblibCallback:
    def __init__(self, pbar: tqdm) -> None:
        self.pbar = pbar
        self.total = getattr(pbar, 'total', None)
        self.delayed_tasks = 0
        self.delayed_err = 0

    def __call__(self, batch_size, n_completed_tasks, elapsed_time, result_lst) -> None:
        result_lst = ifn(result_lst, [])
        info_lst = []
        delayed_tasks = len(result_lst)
        if delayed_tasks != self.delayed_tasks:
            delayed_err = sum(isinstance(result, Exception) for result in result_lst)
            self.delayed_tasks = delayed_tasks
            self.delayed_err = delayed_err

        info_lst.append(f"error: {self.delayed_err}/{self.delayed_tasks}="
                        f"{round(self.delayed_err/self.delayed_tasks*100 if self.delayed_tasks > 0 else 0, 3)}%")
        if self.total is not None:
            remain_time = elapsed_time * (self.total / n_completed_tasks - 1)
            info_lst.append(f"remain_time: {datetime.timedelta(seconds=remain_time)}")

        self.pbar.update(batch_size)
        self.pbar.set_postfix_str(', '.join(info_lst))


class AutoPackUnpack:
    ALL_FEATURES = {'tar', '7z'}
    SUFFIX_PREFERED_LST = ('.7z', '.tar.gz')
    def __init__(self, debug_child=False,
                 required_features: Iterable[str]=ALL_FEATURES,
                 use_gnutar=True) -> None:
        self.debug_child = debug_child
        self.avaliable_features = set()
        self.avaliable_suffixes = set()
        if 'tar' in required_features:
            if use_gnutar:
                try:
                    self.health_check_gnutar()
                except Exception as err:
                    warnings.warn(f"AutoPackUnpack: GNU tar not available!\n"
                                  f"{repr(err)}\n"
                                  f"Fallback to Python built-in tarfile.")
                    self.tar_backend = 'tarfile'
                    self.avaliable_features.add('tar')
                    self.avaliable_suffixes.add('.tar.gz')
                else:
                    self.tar_backend = 'gnutar'
                    self.avaliable_features.add('tar')
                    self.avaliable_suffixes.add('.tar.gz')
        if '7z' in required_features:
            try:
                self.health_check_7z()
            except Exception as err:
                warnings.warn(f"AutoPackUnpack: 7z not available!\n"
                              f"{repr(err)}\n"
                              f"Some scripts may throw exceptions.\n"
                              f"Please try "
                              f"\"apt install p7zip-full\" or \"conda install -c conda-forge p7zip\" "
                              f"to install p7zip.\n")
            else:
                self.avaliable_features.add('7z')
                self.avaliable_suffixes.add('.7z')

        try:
            self.prefer_suffix = next(x for x in self.SUFFIX_PREFERED_LST if x in self.avaliable_suffixes)
        except StopIteration:
            raise Exception("There is no packaging method available!")

    def health_check_gnutar(self) -> None:
        child = pexpect.spawn('tar',  ["--version"], timeout=60)
        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)

    def health_check_7z(self) -> None:
        child = pexpect.spawn('7z',  ["--help"], timeout=60)
        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)

    def auto_pack(self, dn: StrPath, fn: StrPath, rm_fn=True):
        dn = use_path(dir_path=dn, new=False)
        if rm_fn and Path(fn).exists():
            fn.unlink()
        fn = use_path(file_path=fn)

        if tuple(fn.suffixes[-2:]) == ('.tar', '.gz'):
            vin('tar', self.avaliable_features)
            if self.tar_backend == 'tarfile':
                with tarfile.open(fn, "w:gz") as tar:
                    tar.add(dn, arcname='.')
            elif self.tar_backend == 'gnutar':
                child = pexpect.spawn('tar',  ["-czf", fn.absolute().as_posix(), '.'], 
                                      cwd=dn.as_posix(), timeout=None)
                if self.debug_child: child.logfile_read = sys.stdout.buffer
                child_safe_wait(child)
        elif fn.suffix == '.7z':
            vin('7z', self.avaliable_features)
            child = pexpect.spawn('7z',  ["a", fn.absolute().as_posix(), '.'], 
                                    cwd=dn.as_posix(), timeout=None)
            if self.debug_child: child.logfile_read = sys.stdout.buffer
            child_safe_wait(child)
        else:
            raise Exception(f"Unknow suffix: {fn.suffix} or {tuple(fn.suffixes[-2:])}")

    def auto_unpack(self, fn: StrPath, dn: StrPath, rm_dn=False):
        fn = use_path(file_path=fn, new=False)
        if rm_dn and Path(dn).exists():
            shutil.rmtree(dn)
        dn = use_path(dir_path=dn)

        if tuple(fn.suffixes[-2:]) == ('.tar', '.gz'):
            vin('tar', self.avaliable_features)
            if self.tar_backend == 'tarfile':
                with tarfile.open(fn) as f:
                    f.extractall(dn)
            elif self.tar_backend == 'gnutar':
                child = pexpect.spawn('tar',  ["-xzf", fn.absolute().as_posix()], 
                                      cwd=dn.as_posix(), timeout=None)
                if self.debug_child: child.logfile_read = sys.stdout.buffer
                child_safe_wait(child)
        elif fn.suffix == '.7z':
            vin('7z', self.avaliable_features)
            child = pexpect.spawn('7z',  ["x", fn.absolute().as_posix(), '-aoa'], 
                                    cwd=dn.as_posix(), timeout=None)
            if self.debug_child: child.logfile_read = sys.stdout.buffer
            child_safe_wait(child)
        else:
            raise Exception(f"Unknow suffix: {fn.suffix} or {tuple(fn.suffixes[-2:])}")

def EmptyDecorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def rdkit_mol_decorator(func: Callable[[Mol, Any], Any]):
    @wraps(func)
    def wrapper(smi: str, *args, **kwargs) -> Any:
        with rdkit_log_handle() as rdLog:
            try:
                mol: Mol = Chem.MolFromSmiles(smi)
                assert mol is not None
                ret: Any = func(mol, *args, **kwargs)
            except Exception as err:
                raise RDKitException(f"{rdLog()}: {repr(err)}")
        return ret
    return wrapper

def chunked(sequence: Sequence, n: int) -> List[List]:
    """
    Splits a sequence into chunks of size n.
    """
    return [sequence[i:i + n] for i in range(0, len(sequence), n)]
