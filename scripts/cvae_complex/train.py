"""Train CVAE Complex Model

The training is divided into two stages:
    - Encoder pre-training: Use large small molecular data set (ChEMBL)
    - Condition fine-tuning: Use protein-ligand complex data set (CrossDocked2020)
"""

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import esm

from deepblock.datasets import name_to_dataset_cls, \
    ComplexAADataset, ComplexAACollate
from deepblock.model import CVAEComplex
from deepblock.learner import LearnerCVAEComplex
from deepblock.utils import CheckpointManger, Vocab, \
    auto_dump, auto_load, generate_train_id, ifn, init_logging, \
    mix_config, init_random, pretty_kv, use_path, rel_fn_dic

PROJECT_NAME = "DeepBlock"
MODEL_TYPE = "cvae_complex"
STAGE = "train"


def parse_opt():
    parser = argparse.ArgumentParser(description='Train: CVAE Complex')
    parser.add_argument("--base-train-id", type=str)
    parser.add_argument("--base-weight-choice", type=str, default="valid/loss")
    parser.add_argument("--crossdocked-cached-dn", type=str,
                        default="saved/preprocess/crossdocked")
    parser.add_argument("--chembl-cached-dn", type=str,
                        default="saved/preprocess/chembl")
    parser.add_argument("--pdbbind-cached-dn", type=str,
                        default="saved/preprocess/pdbbind")
    parser.add_argument("--include", type=str,
                        choices=name_to_dataset_cls.keys(), default='crossdocked')
    parser.add_argument("--vocab-fn", type=str, default=f"saved/preprocess/merge_vocabs"
                        f"/chembl,crossdocked&frag_vocab.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--esm-device", type=str, default=None)
    parser.add_argument("--esm-cache-device", type=str, default=None)
    parser.add_argument("--no-valid-prior", action="store_true")
    parser.add_argument("--suffix", type=str)
    opt = mix_config(parser, __file__)
    return opt


def resume_bpt(train_id: str, weight_choice: str):
    saved_dn = Path(f"saved/{MODEL_TYPE}/{train_id}")
    weights_dn = use_path(dir_path=saved_dn / "weights")
    best_fn = use_path(file_path=saved_dn / "best.json")
    opt_fn = use_path(file_path=saved_dn / "opt.json")
    bpt = auto_load(opt_fn, to_edict=True)
    ckpt = CheckpointManger(weights_dn, best_fn, save_step=10)
    bpt.base_weight_fn = ckpt.pick(weight_choice).as_posix()
    return bpt


if __name__ == '__main__':
    opt = parse_opt()
    opt.esm_device = ifn(opt.esm_device, opt.device)
    opt.esm_cache_device = ifn(opt.esm_cache_device, opt.device)

    # Define Path
    opt.train_id = generate_train_id(opt.dev, opt.suffix)
    saved_dn = Path(f"saved/{MODEL_TYPE}/{opt.train_id}")
    log_fn = use_path(file_path=saved_dn / f"{STAGE}.log")
    weights_dn = use_path(dir_path=saved_dn / "weights")
    best_fn = use_path(file_path=saved_dn / "best.json")
    opt_fn = use_path(file_path=saved_dn / "opt.json")

    train_smi_fn = use_path(file_path=saved_dn / "train_smi.json")
    valid_smi_fn = use_path(file_path=saved_dn / "valid_smi.json")

    if opt.base_train_id is not None:
        bpt = resume_bpt(opt.base_train_id, opt.base_weight_choice)
        override_args = (
            'base_weight_fn',
            'random_seed', 'vocab_fn',
            'x_max_len', 'c_max_len', 'split_pro',
            'esm_model_name',
            *CVAEComplex.opt_args,
            # *LearnerCVAEComplex.opt_args,
        )
        for name in override_args:
            if name not in opt:
                print(f"Keep old value: {bpt[name]=}")
                opt[name] = bpt[name]
            elif name not in bpt:
                print(f"Find new value: {opt[name]=}")
            elif opt[name] != bpt[name]:
                print(f"Notice different value: {bpt[name]=} {opt[name]=}")

    # Initialize log, device, config, wandb
    init_logging(log_fn)
    logging.info(f'opt: {opt}')
    device = torch.device(opt.device)
    esm_device = torch.device(opt.esm_device)
    esm_cache_device = torch.device(opt.esm_cache_device)
    auto_dump(dict(opt), opt_fn, json_indent=True)
    wandb.init(
        project=f"{PROJECT_NAME}-{MODEL_TYPE}",
        name=opt.train_id,
        config=opt,
        mode="disabled" if (opt.dev or opt.no_wandb) else "online"
    )

    # Reproducibility
    init_random(opt.random_seed)

    # Dataset
    _dataset = name_to_dataset_cls[opt.include](
        opt[f"{opt.include}_cached_dn"])
    if opt.vocab_fn is not None:
        x_vocab = Vocab(**auto_load(opt.vocab_fn))
        opt.vocab_fn = saved_dn / "x_vocab.json"
        auto_dump(x_vocab.to_dict(), opt.vocab_fn)
        logging.info(f"Backup x_vocab to {opt.vocab_fn}")
    else:
        opt.vocab_fn = saved_dn / "x_vocab.json"
        x_vocab = Vocab(**auto_load(opt.vocab_fn))

    rel_fn = rel_fn_dic[opt.rel_fn_type]
    _dataset_opt = dict(d=_dataset,
                        rel_fn=rel_fn,
                        x_vocab=x_vocab,
                        x_max_len=opt.x_max_len,
                        c_max_len=opt.c_max_len,
                        split_pro_dic=opt.split_pro,
                        is_dev=opt.dev)
    train_set = ComplexAADataset(**_dataset_opt, split_key='train')
    valid_set = ComplexAADataset(**_dataset_opt, split_key='valid')

    logging.info(f"len(train_set): {len(train_set)}, len(valid_set): {len(valid_set)}")
    logging.info(f"train_set.cleaning_status: {pretty_kv(train_set.cleaning_status)}")
    logging.info(f"valid_set.cleaning_status: {pretty_kv(valid_set.cleaning_status)}")
    
    auto_dump(train_set.id_smi_dic, train_smi_fn)
    auto_dump(valid_set.id_smi_dic, valid_smi_fn)
    logging.info(f"Backup smi to {train_smi_fn}, {valid_smi_fn}")

    # Load ESM-2 model
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(
        opt.esm_model_name)

    # Dataloader
    collate_fn = ComplexAACollate(x_vocab.special_idx, esm_alphabet)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=opt.num_workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=opt.valid_batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=opt.num_workers)
    if opt.use_esm_warm and opt.esm_cache_enable:
        warm_loader = DataLoader(dataset=ConcatDataset((train_set, valid_set)), batch_size=opt.warm_batch_size,
                                 shuffle=False, collate_fn=collate_fn, num_workers=opt.num_workers)

    # Model
    model = CVAEComplex(x_input_dim=len(train_set.x_vocab),
                        c_input_dim=esm_model.embed_dim,
                        **{k: opt[k] for k in CVAEComplex.opt_args})
    if 'base_weight_fn' in opt:
        logging.info(f"base_weight_fn: {opt.base_weight_fn}")
        model.load_state_dict(torch.load(
            opt.base_weight_fn, map_location=torch.device('cpu')))
    if not model.c_attend:
        logging.warning("Notice: No c_attend!")

    # Checkpoint
    ckpt = CheckpointManger(weights_dn, best_fn, save_step=10)

    # Learner
    learner = LearnerCVAEComplex(model, esm_model, device, esm_device, esm_cache_device, esm_alphabet,
                                 x_vocab=x_vocab, collate_fn=collate_fn, rel_fn=rel_fn,
                                 **{k: opt[k] for k in LearnerCVAEComplex.opt_args})

    # Warm ESM cache
    if opt.use_esm_warm and opt.esm_cache_enable:
        learner.warm_esm(warm_loader)
        logging.info("Warm up ESM done!")
        esm_cache_status = learner.get_esm_cache_status()
        logging.info(pretty_kv(esm_cache_status,
                    ndigits=5, prefix=f'ESM cache status: '))
        if opt.move_esm_model_to_cpu_after_warm:
            learner.esm_model.cpu()
            torch.cuda.empty_cache()
            logging.info("Move ESM model to cpu after warm done!")

    # Loop
    for epoch in range(opt.max_epochs):
        print(f'epoch -> {epoch}/{opt.max_epochs - 1}, train_id -> {opt.train_id}')
        loss_dic, other_dic = {}, {}

        desc = 'Train_TF'
        loss_item = learner.train_tf(train_loader, desc=desc)
        logging.info(pretty_kv(
            dict(epoch=epoch, **asdict(loss_item)), ndigits=5, prefix=f'{desc}: '))
        loss_dic.update({
            f'train/recons': loss_item.recons_loss,
            f'train/kld': loss_item.kld_loss,
            f'train/bow': loss_item.bow_loss,
            f'train/rel': loss_item.rel_loss,
            f'train/loss': loss_item.loss
        })
        other_dic.update(
            {f'hyper/{k}': v for k, v in learner.hyper_dic.items()})

        desc = 'Valid_TF'
        loss_item = learner.eval_tf(valid_loader, desc=desc)
        logging.info(pretty_kv(
            dict(epoch=epoch, **asdict(loss_item)), ndigits=5, prefix=f'{desc}: '))
        loss_dic.update({
            f'valid/recons': loss_item.recons_loss,
            f'valid/kld': loss_item.kld_loss,
            f'valid/bow': loss_item.bow_loss,
            f'valid/rel': loss_item.rel_loss,
            f'valid/loss': loss_item.loss
        })

        if not opt.no_valid_prior:
            desc = 'Valid_TF_Prior'
            loss_item = learner.eval_tf(valid_loader, desc=desc, use_prior=True)
            logging.info(pretty_kv(
                dict(epoch=epoch, **asdict(loss_item)), ndigits=5, prefix=f'{desc}: '))
            loss_dic.update({
                f'valid/prior_recons': loss_item.recons_loss,
                f'valid/prior_bow': loss_item.bow_loss,
            })

        desc = 'Valid'
        error = learner.eval(valid_loader, desc=desc)
        logging.info(pretty_kv(dict(epoch=epoch, error=error),
                     ndigits=5, prefix=f'{desc}: '))
        loss_dic.update({
            f'valid/error': error
        })

        if not opt.no_valid_prior:
            desc = 'Valid_Prior'
            error = learner.eval(valid_loader, desc=desc, use_prior=True)
            logging.info(pretty_kv(dict(epoch=epoch, error=error),
                        ndigits=5, prefix=f'{desc}: '))
            loss_dic.update({
                f'valid/prior_error': error
            })

        dec_dic, weight_fn = ckpt.submit(
            epoch=epoch, values=loss_dic, weight=model.state_dict())
        if len(dec_dic) > 0:
            logging.info(pretty_kv(dec_dic, prefix=f'Save to {weight_fn}: '))
        wandb.log({**loss_dic, **other_dic})

    logging.info("Done!")
