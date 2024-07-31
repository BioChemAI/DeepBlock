from collections import OrderedDict
from dataclasses import asdict
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Sequence, Tuple
import warnings
import torch
from torch import Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from halo import Halo
from easydict import EasyDict as edict
import esm
from esm.model.esm2 import ESM2
from esm.data import Alphabet

from ..datasets import ComplexAABatch, ComplexAAItem, ComplexAACollate
from ..utils import BetaVAEScheduler, CVAELossItem, CVAELossList, \
    CheckpointManger, StrPath, TRelFn, Vocab, \
    auto_load, convert_bytes, ifn, pretty_kv, use_path, rel_fn_dic
from ..model import CVAEComplex

class LearnerCVAEComplex():
    opt_args = (
        'optim_lr', 'optim_lr_scheduler', 
        'x_max_len', 'c_max_len', 
        'clip_max_norm', 'show_tqdm', 'beta_vae', 'bow_weight', 'rel_weight',
        'esm_cache_enable'
    )
    
    def __init__(self, model: CVAEComplex, esm_model: ESM2, 
                device: torch.device, esm_device: torch.device, esm_cache_device: torch.device,
                esm_alphabet: Alphabet,
                optim_lr: float, optim_lr_scheduler: Dict, 
                x_vocab: Vocab, collate_fn: ComplexAACollate, rel_fn: TRelFn,
                x_max_len: int, c_max_len: int, 
                clip_max_norm: float=None, show_tqdm: bool=True,
                beta_vae: Dict={}, bow_weight: Number=1, rel_weight: Number=1,
                esm_cache_enable: bool=False,
                inference_only: bool=False, upstream_opt: edict=edict()) -> None:
        self.model = model.to(device)
        self.esm_model = esm_model.to(esm_device)

        self.device = device
        self.esm_device = esm_device
        self.esm_cache_device = esm_cache_device
        self.esm_alphabet = esm_alphabet
        self.optimizer = optim.Adam(self.model.parameters(), lr=optim_lr)
        _LR = getattr(optim.lr_scheduler, optim_lr_scheduler.mode)
        self.scheduler = _LR(self.optimizer, **optim_lr_scheduler.value)

        self.x_vocab = x_vocab
        self.collate_fn = collate_fn
        self.rel_fn = rel_fn
        self.x_max_len = x_max_len
        self.c_max_len = c_max_len

        self.clip_max_norm = clip_max_norm
        self.tqdm_disable = not show_tqdm
        self.beta_vae_scheduler = BetaVAEScheduler(**beta_vae)

        self.bow_weight = bow_weight
        self.rel_weight = rel_weight

        self.esm_cache_enable = esm_cache_enable
        self.esm_cache: Dict[Hashable, Tensor] = dict()

        self.inference_only = inference_only
        self.upstream_opt = upstream_opt

        self.hyper_dic = dict()

    def _beyond_inference(self):
        if self.inference_only:
            raise Exception("The current initialization method only supports inference")

    def _esm_inference(self, c: Tensor):
        c = c.to(self.esm_device)
        with torch.no_grad():
            esm_results = self.esm_model(c, repr_layers=[self.esm_model.num_layers])
        c_repr: Tensor = esm_results["representations"][self.esm_model.num_layers]
        c_repr = c_repr.detach()
        return c_repr

    def _esm_inference_batch(self, batch: ComplexAABatch):
        if self.esm_cache_enable:
            # Find c_hash without cache
            no_cache_idx_in_batch_lst = [idx for idx, _c_hash in enumerate(batch.c_hash) if _c_hash not in self.esm_cache]
            if len(no_cache_idx_in_batch_lst) > 0:
                no_cache_c = batch.c[no_cache_idx_in_batch_lst]
                no_cache_repr = self._esm_inference(no_cache_c)
                for idx_in_tensor, idx_in_batch in enumerate(no_cache_idx_in_batch_lst):
                    repaired_tensor = no_cache_repr[idx_in_tensor, :batch.c_len[idx_in_batch], :]
                    self.esm_cache[batch.c_hash[idx_in_batch]] = repaired_tensor.to(self.esm_cache_device)
            repr_lst = [self.esm_cache[_c_hash] for _c_hash in batch.c_hash]
            c_repr = pad_sequence(repr_lst, padding_value=0, batch_first=True)
        else:
            c_repr = self._esm_inference(batch.c)
        return c_repr

    def get_esm_cache_status(self):
        f_ele_bytes: Callable[[Tensor], int] = lambda t: t.element_size() * t.nelement()
        amount = len(self.esm_cache)
        nbytes = sum(v.storage().nbytes() for v in self.esm_cache.values())
        ele_bytes = sum(f_ele_bytes(v) for v in self.esm_cache.values())
        total_len = sum(len(v) for v in self.esm_cache.values())
        return dict(amount=amount, nbytes=nbytes, ele_bytes=ele_bytes, size=convert_bytes(ele_bytes), total_len=total_len)

    def warm_esm(self, loader: Sequence[ComplexAABatch], desc='Warm_ESM'):
        self.esm_model.eval()

        pbar = tqdm(loader, desc=desc, disable=self.tqdm_disable)
        for batch in pbar:
            if batch.c is not None:
                self._esm_inference_batch(batch)

    def train_tf(self, loader: Sequence[ComplexAABatch], desc='Train_TF') -> CVAELossItem:
        self._beyond_inference()
        self.model.train()
        self.esm_model.eval()

        self.hyper_dic['lr'] = self.scheduler.get_last_lr()[0]
        self.hyper_dic['beta'] = self.beta_vae_scheduler.get_last_beta()
        print(pretty_kv(self.hyper_dic, ndigits=5, prefix='scheduler -> '))

        loss_lst = CVAELossList()
        pbar = tqdm(loader, desc=desc, disable=self.tqdm_disable)
        for batch in pbar:
            batch.x = batch.x.to(self.device)

            self.optimizer.zero_grad()

            if batch.c is not None:
                c_repr = self._esm_inference_batch(batch).to(self.device)
            else:
                c_repr = None
            
            (recon, bow_logits, c_attn, 
                recog_mu, recog_log_var, prior_mu, prior_log_var
            ) = self.model(
                x=batch.x, x_len=batch.x_len,
                c=c_repr, c_len=batch.c_len,
                # variational=False # !!!! TEST ONLY !!!!
            )

            recons_loss = self.model.recons_loss(recon, batch.x, self.x_vocab.special_idx)
            if batch.c is not None:
                kld_loss = self.model.kld_loss(
                    (recog_mu, prior_mu), (recog_log_var, prior_log_var))
            else:
                kld_loss = self.model.kld_loss(
                    (recog_mu, torch.tensor(0)), (recog_log_var, torch.tensor(1).log()))
                
            bow_loss = self.model.bow_loss(bow_logits, batch.x, self.x_vocab.special_idx)
            if batch.c_rel is not None and self.model.c_attend:
                rel_loss = self.model.rel_loss(c_attn, batch.c_rel, batch.c_len)
            else:
                rel_loss = torch.tensor(0)
            
            beta = self.beta_vae_scheduler.get_last_beta()
            loss = recons_loss + kld_loss * beta + bow_loss * self.bow_weight + rel_loss * self.rel_weight

            loss_batch = loss / batch.total
            loss_batch.backward()

            if self.clip_max_norm:
                clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            self.optimizer.step()

            loss_item = CVAELossItem(recons_loss.item(), kld_loss.item(), loss.item(), batch.total, 
                                     bow_loss=bow_loss.item(), rel_loss=rel_loss.item())
            loss_lst.append(loss_item)

            pbar.set_postfix_str(pretty_kv(asdict(loss_item.mean())))

        self.scheduler.step()
        self.beta_vae_scheduler.step()

        return loss_lst.mean()


    def eval_tf(self, loader: Sequence[ComplexAABatch], desc='Eval_TF', use_prior=False) -> CVAELossItem:
        self.model.eval()
        self.esm_model.eval()

        loss_lst = CVAELossList()
        pbar = tqdm(loader, desc=desc, disable=True)
        for batch in pbar:
            # Hybrid training support
            if batch.c is None and use_prior:
                continue

            batch.x = batch.x.to(self.device)

            if batch.c is not None:
                c_repr = self._esm_inference_batch(batch).to(self.device)
            else:
                c_repr = None

            with torch.no_grad():
                (recon, bow_logits, c_attn, 
                    recog_mu, recog_log_var, prior_mu, prior_log_var
                ) = self.model(
                    x=batch.x, x_len=batch.x_len,
                    c=c_repr, c_len=batch.c_len,
                    variational=False, use_prior=use_prior
                )

            recons_loss = self.model.recons_loss(recon, batch.x, self.x_vocab.special_idx)
            if batch.c is not None:
                kld_loss = self.model.kld_loss(
                    (recog_mu, prior_mu), (recog_log_var, prior_log_var))
            else:
                kld_loss = self.model.kld_loss(
                    (recog_mu, torch.tensor(0)), (recog_log_var, torch.tensor(1).log()))
                
            bow_loss = self.model.bow_loss(bow_logits, batch.x, self.x_vocab.special_idx)
            if batch.c_rel is not None and self.model.c_attend:
                rel_loss = self.model.rel_loss(c_attn, batch.c_rel, batch.c_len)
            else:
                rel_loss = torch.tensor(0)
            
            beta = self.beta_vae_scheduler.get_last_beta()
            loss = recons_loss + kld_loss * beta + bow_loss * self.bow_weight + rel_loss * self.rel_weight

            loss_item = CVAELossItem(recons_loss.item(), kld_loss.item(), loss.item(), batch.total, 
                                     bow_loss=bow_loss.item(), rel_loss=rel_loss.item())
            loss_lst.append(loss_item)

            pbar.set_postfix_str(pretty_kv(asdict(loss_item.mean())))

        return loss_lst.mean()

    def eval(self, loader: Sequence[ComplexAABatch], desc='Eval', use_prior=False) -> float:
        self.model.eval()
        self.esm_model.eval()

        seq_tup_lst: List[Tuple[List[str], List[str]]] = []
        pbar = tqdm(loader, desc=desc, disable=False)
        for batch in pbar:
            # Hybrid training support
            if batch.c is None and use_prior:
                continue
            
            batch.x = batch.x.to(self.device)

            if batch.c is not None:
                c_repr = self._esm_inference_batch(batch).to(self.device)
            else:
                c_repr = None

            with torch.no_grad():
                (recon, bow_logits, c_attn, 
                    recog_mu, recog_log_var, prior_mu, prior_log_var, 
                    prd, prd_len
                ) = self.model(
                    x=batch.x, x_len=batch.x_len,
                    c=c_repr, c_len=batch.c_len,
                    special_idx=self.x_vocab.special_idx, x_max_len=self.x_max_len, 
                    tf=False, variational=False, use_prior=use_prior
                )

            # Moving to the CPU as a whole may accelerate (not implemented)
            for i in range(batch.total):
                prd_seq = prd[i][:prd_len[i]].tolist()
                x_seq = batch.x[i][1:batch.x_len[i]].tolist()
                seq_tup_lst.append((prd_seq, x_seq))

        equal_lst = [x[0] == x[1] for x in seq_tup_lst]
        error = 1 - sum(equal_lst) / len(equal_lst)
        return error

    def sample(self, item: ComplexAAItem, 
               batch_size: int, desc='Sample', variational=True,
               use_groundtruth_rel=False,
               use_force_mean=False):
        batch = self.collate_fn([item])

        self.model.eval()
        self.esm_model.eval()

        # pbar = tqdm(loader, desc=desc, disable=False)

        assert batch.c is not None, "How can I predict without protein?"
        c_repr = self._esm_inference_batch(batch).to(self.device)
        c_rel = batch.c_rel.float().to(self.device) if batch.c_rel is not None else None

        while True:
            with torch.no_grad():
                (recon, c_attn, 
                    prior_mu, prior_log_var, 
                    prd, prd_len
                ) = self.model.sample(
                    c=c_repr, c_len=batch.c_len,
                    special_idx=self.x_vocab.special_idx, x_max_len=self.x_max_len, 
                    rep_times=batch_size, variational=variational,
                    use_groundtruth_rel=use_groundtruth_rel, c_rel=c_rel,
                    use_force_mean=use_force_mean
                )

            if c_attn is not None:
                c_attn = c_attn.cpu().detach().numpy()
            for i in range(batch_size):
                prd_seq = prd[0, i][:prd_len[0, i]].tolist()
                yield prd_seq, c_attn

    def optimize(self, batch: ComplexAABatch, 
                 desc='Optimize', variational=True) -> 'OrderedDict[Hashable, edict]':
        """Optimizing molecules for targets
        """
        self.model.eval()
        self.esm_model.eval()

        result_od = OrderedDict()

        batch.x = batch.x.to(self.device)

        if batch.c is not None:
            c_repr = self._esm_inference_batch(batch).to(self.device)
        else:
            warnings.warn("Unusual Optimization Task: Unconditional")
            c_repr = None

        with torch.no_grad():
            (recon, bow_logits, c_attn, 
                recog_mu, recog_log_var, prior_mu, prior_log_var, 
                prd, prd_len
            ) = self.model(
                x=batch.x, x_len=batch.x_len,
                c=c_repr, c_len=batch.c_len,
                special_idx=self.x_vocab.special_idx, x_max_len=self.x_max_len, 
                tf=False, variational=variational, use_prior=False
            )

        c_attn = c_attn.cpu().detach().numpy()
        for i in range(batch.total):
            _dic = edict()
            _dic.prd_seq = prd[i][:prd_len[i]].tolist()
            _dic.x_seq = batch.x[i][1:batch.x_len[i]].tolist()
            _dic.c_attn = c_attn[i]
            _dic.is_same = _dic.prd_seq == _dic.x_seq
            result_od[batch.id[i]] = _dic

        return result_od

    def encode_emb(self, batch: ComplexAABatch, 
                   desc='Encode embedding') -> Tensor:
        self.model.eval()
        self.esm_model.eval()

        batch.x = batch.x.to(self.device)
        c_repr = self._esm_inference_batch(batch).to(self.device)

        with torch.no_grad():
            x_state = self.model.encode_x(batch.x, batch.x_len)
            c_state, c_attn = self.model.encode_c(c_repr, batch.c_len)
            xc_state = torch.cat((x_state, c_state), dim=1)
            recog_mu, recog_log_var = self.model.recog_head(xc_state)
        return recog_mu

    def decode_emb(self, z: Tensor, batch: ComplexAABatch, 
                   desc='Decode embedding') -> list[int]:
        self.model.eval()
        self.esm_model.eval()

        z = z.to(self.device)
        c_repr = self._esm_inference_batch(batch).to(self.device)

        with torch.no_grad():
            c_state, c_attn = self.model.encode_c(c_repr, batch.c_len)
            z = torch.cat((z, c_state), dim=1)
            recon, prd, prd_len = self.model.decode(z, self.x_vocab.special_idx, self.x_max_len)
        prd_seqs = [prd[i][:prd_len[i]].tolist() for i in range(z.shape[0])]
        return prd_seqs

    @classmethod
    def init_inference_from_saved_dn(cls, 
            saved_dn: StrPath, weight_choice: str,
            device: str="cpu", esm_device: str=None, esm_cache_device: str=None,
            **kwargs) -> 'LearnerCVAEComplex':
        
        """Recover learner from saved_dn

        Parameters used from opt:
        esm_model_name, x_max_len, c_max_len, rel_fn_type, CVAEComplex.opt_args
        """

        saved_dn = use_path(dir_path=saved_dn, new=False)
        desc = f'Initialize {cls.__name__} from {saved_dn.name}'
        spinner = Halo(text=desc)
        spinner.start()

        # Device
        esm_device = ifn(esm_device, device)
        esm_cache_device = ifn(esm_cache_device, device)

        spinner.text = desc + (f' -> device: {device}, esm_device: {esm_device}, '
                              f'esm_cache_device: {esm_cache_device}')
        device = torch.device(device)
        esm_device = torch.device(esm_device)
        esm_cache_device = torch.device(esm_cache_device)

        # Option
        opt_fn = use_path(file_path=saved_dn / "opt.json", new=False)
        opt = auto_load(opt_fn, to_edict=True)

        # Weight
        weights_dn = use_path(dir_path=saved_dn / "weights", new=False)
        best_fn = use_path(file_path=saved_dn / "best.json", new=False)
        ckpt = CheckpointManger(weights_dn, best_fn, save_step=10)
        opt.base_weight_fn = ckpt.pick(weight_choice).as_posix()

        # Vocab
        opt.vocab_fn = use_path(dir_path=saved_dn / "x_vocab.json", new=False)
        spinner.text = desc + f" -> vocab_fn: {opt.vocab_fn}"
        x_vocab = Vocab(**auto_load(opt.vocab_fn))

        # ESM-2 model
        spinner.text = desc + f" -> esm_model_name: {opt.esm_model_name}"
        esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(
            opt.esm_model_name)
        
        # Data function
        collate_fn = ComplexAACollate(x_vocab.special_idx, esm_alphabet)
        rel_fn = rel_fn_dic[opt.rel_fn_type]

        # Model
        spinner.text = desc + f" -> weight_fn: {Path(opt.base_weight_fn).name}"
        model = CVAEComplex(x_input_dim=len(x_vocab),
                            c_input_dim=esm_model.embed_dim,
                            **{k: opt[k] for k in CVAEComplex.opt_args})
        model.load_state_dict(torch.load(
            opt.base_weight_fn, map_location=torch.device('cpu')))

        obj = cls(model, esm_model, 
                  device, esm_device, esm_cache_device, esm_alphabet, 
                  x_vocab=x_vocab, collate_fn=collate_fn, rel_fn=rel_fn,
                  inference_only=True, upstream_opt=opt,
                  **{k: opt[k] for k in cls.opt_args if k not in kwargs},
                  **kwargs)
        
        spinner.succeed(desc + f" -> Done!")
        return obj
    