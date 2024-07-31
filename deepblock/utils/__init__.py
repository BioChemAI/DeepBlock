"""High level functions provided by DeepBlock for handling exceptions, 
data, algorithms, logs, visualization, biochemistry, etc.
"""

from .common import *
from .vocab import Vocab, VocabSpecialIndex, VocabSpecialSymbol
from .time_limit import time_limit
from . import complex_to_aa, mol_to_frag, smi_to_fp, smi_to_word
# from . import sparse_gp

__all__ = [
    "Mol", "StrPath",
    "use_path", "use_memory",
    "auto_dump_load_suffixes", "auto_dump", "auto_load", "auto_loadm",
    "rdkit_log_handle", "ob_log_handle",
    "init_logging",
    "unique_by_key",
    "Toc",
    "split_pro_to_idx",
    "mix_config",
    "pretty_kv", "summary_arr",
    "init_random", "sorted_seqs", "pick_by_idx",
    "get_time_str", "generate_train_id",
    "VAELossItem", "CVAELossItem", "VAELossList", "CVAELossList",
    "BetaVAEScheduler", "CheckpointManger",
    "sha256_dic", "sequential_file_hash",
    "convert_bytes", "get_file_size", "file_hash",
    "pretty_dataclass_repr",
    "norm_rel_fn", "exp_rel_fn", "rel_fn_dic",
    "gc_disabled",
    "assert_prob",
    "ifn", "vin",
    "download_pbar",
    "ChildUnexpectedTerminationException", "child_safe_wait",
    "chmod_plus_x",
    "ignore_exception", "return_exception", "delayed_return_exception",
    "hook_joblib", "LoggingTrackerJoblibCallback", "TqdmTrackerJoblibCallback",
    "AutoPackUnpack", 
    'EmptyDecorator', 
    'rdkit_mol_decorator',
    
    "Vocab", "VocabSpecialIndex", "VocabSpecialSymbol",
    "time_limit",

    "complex_to_aa", "mol_to_frag", "smi_to_fp", "smi_to_word",

    "gaussian_process", "chunked"
]
