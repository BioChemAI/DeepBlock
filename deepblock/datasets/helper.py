from typing import Dict, Tuple, Type, Union
from . import RawThirdDataset, ChEMBLDataset, CrossDockedDataset, PDBbindDataset, \
    ComplexAADataset, ComplexAAItem

name_to_dataset_cls: Dict[str, Type[RawThirdDataset]] = {
    "chembl": ChEMBLDataset,
    "crossdocked": CrossDockedDataset,
    "pdbbind": PDBbindDataset,
}

def get_complex_aa_dataset_by_learner(name: str, cached_dn: str, 
                                      split_key: str, learner: Type):
    split_pro_dic=learner.upstream_opt.split_pro
    _dataset = name_to_dataset_cls[name](cached_dn)
    _dataset_opt = dict(d=_dataset,
                        rel_fn=learner.rel_fn,
                        x_vocab=learner.x_vocab,
                        x_max_len=learner.x_max_len,
                        c_max_len=learner.c_max_len,
                        split_pro_dic=split_pro_dic,
                        is_dev=False)
    _set = ComplexAADataset(**_dataset_opt, split_key=split_key)
    return _set

def get_complex_aa_item_by_learner(name: str, cached_dn: str, 
                                   split_key: str, learner: Type, 
                                   cid: str, 
                                   return_tup: bool=False
                                   ) -> Union[ComplexAAItem, Tuple[ComplexAAItem, ...]]:
    split_pro_dic=learner.upstream_opt.split_pro
    _dataset = name_to_dataset_cls[name](cached_dn)
    _dataset_opt = dict(d=_dataset,
                        rel_fn=learner.rel_fn,
                        x_vocab=learner.x_vocab,
                        x_max_len=learner.x_max_len,
                        c_max_len=learner.c_max_len,
                        split_pro_dic=split_pro_dic,
                        is_dev=False)
    _set = ComplexAADataset(**_dataset_opt, split_key=split_key)
    item = _set[_set.id_to_idx(cid)]

    if not return_tup:
        return item
    else:
        meta_lst = _dataset.source("meta")
        meta = next(x for x in meta_lst if x['id'] == cid)
        complex_to_aa_dic = _dataset.source("complex_to_aa")
        aa = complex_to_aa_dic[cid]
        return item, meta, aa
