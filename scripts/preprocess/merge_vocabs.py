"""Merge vocabs
"""

import argparse
from functools import reduce
import logging
from typing import List
from deepblock.datasets import name_to_dataset_cls
from deepblock.utils import auto_dump, init_logging, mix_config, Vocab, summary_arr, use_path

def parse_opt():
    parser = argparse.ArgumentParser(description='Preprocess: Merge vocabs')
    parser.add_argument("--crossdocked-cached-dn", type=str, default="saved/preprocess/crossdocked")
    parser.add_argument("--chembl-cached-dn", type=str, default="saved/preprocess/chembl")
    parser.add_argument("--pdbbind-cached-dn", type=str, default="saved/preprocess/pdbbind")
    parser.add_argument("--includes", nargs='+', type=str, default=['chembl', 'crossdocked'])
    parser.add_argument("--source", type=str, default="frag_vocab")
    parser.add_argument("--output-dn", type=str, default="saved/preprocess/merge_vocabs")
    opt = mix_config(parser, __file__)
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    init_logging(f"{opt.output_dn}/preprocess.log")
    logging.info(opt)

    vocab_lst: List[Vocab] = []
    for name in opt.includes:
        _dataset = name_to_dataset_cls[name](opt[f"{name}_cached_dn"])
        _vocab: Vocab = _dataset.source(opt.source)
        logging.info(f"Loaded! {name}_vocab")
        logging.info(f"{summary_arr(_vocab, key=len)}")
        assert isinstance(_vocab, Vocab), f"{opt.source} for {name} is not a Vocab"
        vocab_lst.append(_vocab)

    vocab = reduce(lambda x, y: x + y, vocab_lst)
    fn = use_path(dir_path=opt.output_dn) / f"{','.join(opt.includes)}&{opt.source}.json"
    auto_dump(vocab.to_dict(), fn)

    logging.info(f"Saved! vocab -> {fn}")
    logging.info(f"{summary_arr(vocab, key=len)}")

    logging.info("Done!")
