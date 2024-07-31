from typing import Callable, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from dataclasses import astuple

from ..utils import VocabSpecialIndex, assert_prob
from . import EncoderRNN, DecoderRNN

class MLPHead(nn.Module):
    """Simple MLP Head
    """
    def __init__(self, input_dim: int, hid_dim: int, output_dim: int, dropout=0.25):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim)
        )
    
    def forward(self, x: Tensor):
        x = self.layers(x)
        return x


class VariationalMLPHead(nn.Module):
    """Variational MLP Head for Recognition Network and Prior Network
    """
    def __init__(self, state_dim, hid_dim, latent_dim, dropout=0.25):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mlp = MLPHead(state_dim, hid_dim, latent_dim*2, dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.mlp(x)
        mu, log_var = x.split(self.latent_dim, dim=1)
        return mu, log_var


class CVAEComplex(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder) for complex
    """
    opt_args = (
        'x_emb_dim',
        'x_enc_hid_dim', 'x_enc_hid_layers', 
        'x_dec_hid_dim', 'x_dec_hid_layers', 
        'c_emb_dim', 'c_eng_hid_dim', # 'c_input_dim',
        'h_hid_dim', 'latent_dim', 'bow_hid_dim', 
        'x_dropout', 'x_bidirectional', # 'x_emb_weight'
        'c_dropout', 'h_dropout', 'bow_dropout',
        'c_attend'
    )
    
    def __init__(self, 
                x_input_dim: int, x_emb_dim: int,
                x_enc_hid_dim: int, x_enc_hid_layers: int,
                x_dec_hid_dim: int, x_dec_hid_layers: int,
                c_input_dim: int, c_emb_dim: int, c_eng_hid_dim: int,
                h_hid_dim: int, latent_dim: int, bow_hid_dim: int,
                x_dropout=0.25, x_bidirectional=False, x_emb_weight: Tensor=None,
                c_dropout=0.25, h_dropout=0.25, bow_dropout=0.25,
                c_attend=True):
        super().__init__()
        
        bid_num = 2 if x_bidirectional else 1
        if x_emb_weight:
            assert x_input_dim, x_emb_dim == x_emb_weight.shape
        self.x_encoder = EncoderRNN(x_input_dim, x_emb_dim, x_enc_hid_dim, x_enc_hid_layers, 
            x_dropout, x_bidirectional, x_emb_weight)

        self.c_emb_dim = c_emb_dim
        self.latent_dim = latent_dim
        self.c_energy = MLPHead(c_input_dim, c_eng_hid_dim, 1, c_dropout)
        self.fc_c_downsample = nn.Linear(c_input_dim, c_emb_dim)

        self.recog_head = VariationalMLPHead(c_emb_dim + bid_num*x_enc_hid_dim*x_enc_hid_layers, 
            h_hid_dim, latent_dim, h_dropout)
        self.prior_head = VariationalMLPHead(c_emb_dim, h_hid_dim, latent_dim, h_dropout)

        self.fc_latent_upsample = nn.Linear(latent_dim + c_emb_dim, x_dec_hid_dim * x_dec_hid_layers)
        self.x_decoder = DecoderRNN(x_input_dim, x_emb_dim, x_dec_hid_dim, x_dec_hid_layers, x_dropout)

        self.bow_head = MLPHead(latent_dim + c_emb_dim, bow_hid_dim, x_input_dim, bow_dropout)

        self.c_attend = c_attend

    def forward(self, x: Tensor, x_len: Tensor, 
                c: Tensor=None, c_len: Tensor=None,
                special_idx: VocabSpecialIndex=None, x_max_len: int=None,
                tf=True, variational=True, use_prior=False):

        # Need to deal with the situation without c
        if c is None and use_prior:
            raise Exception("Cannot use prior networks without conditions")

        bs, device = x.shape[0], x.device
        # x = [bs, x_max_len]
        # c = [bs, c_max_len, c_emb_dim]

        x_state = self.encode_x(x, x_len)
        if c is not None:
            c_state, c_attn = self.encode_c(c, c_len)
        else:
            c_state, c_attn = torch.zeros(bs, self.c_emb_dim, device=device), None
    
        # Drop c_state in pretrain stage
        # x_state = [bs, bid_num*x_enc_hid_dim*x_enc_hid_layers]
        # c_state = [bs, c_emb_dim]

        xc_state = torch.cat((x_state, c_state), dim=1)
        # xc_state = [bs, c_emb_dim+bid_num*x_enc_hid_dim*x_enc_hid_layers]

        recog_mu, recog_log_var = self.recog_head(xc_state)
        if c is not None:
            prior_mu, prior_log_var = self.prior_head(c_state)
        else:
            prior_mu, prior_log_var = None, None
        # (above) = [bs, latent_dim]

        if variational:
            if use_prior:
                z = self.reparametrize(prior_mu, prior_log_var)
            else:
                z = self.reparametrize(recog_mu, recog_log_var)
        else:
            if use_prior:
                z = prior_mu
            else:
                z = recog_mu
        # z = [bs, latent_dim]

        z = torch.cat((z, c_state), dim=1)
        # z = [bs, latent_dim+c_emb_dim]

        bow_logits = self.bow_head(z)
        # bow_logits = [bs, x_output_dim]

        if tf:
            recon = self.decode_tf(z, x, x_len)
            return (recon, bow_logits, c_attn, 
                    recog_mu, recog_log_var, prior_mu, prior_log_var)
        else:
            recon, prd, prd_len = self.decode(z, special_idx, x_max_len)
            return (recon, bow_logits, c_attn, 
                    recog_mu, recog_log_var, prior_mu, prior_log_var, 
                    prd, prd_len)

    def sample(self, c: Tensor, c_len: Tensor,
                special_idx: VocabSpecialIndex, x_max_len: int,
                rep_times: int=1, variational=True,
                use_groundtruth_rel=False, c_rel: Tensor=None,
                use_force_mean=False):
        
        if variational:
            assert rep_times >= 1 and isinstance(rep_times, int)
        else:
            assert rep_times == 1

        bs, device = c.shape[0], c.device
        # c = [bs, c_max_len, c_emb_dim]

        assert not (use_groundtruth_rel and use_force_mean), \
            "Args: use_groundtruth_rel and use_force_mean are mutually exclusive"
        
        if use_groundtruth_rel or use_force_mean:
            _c_attend_last = self.c_attend
            self.c_attend = False
        if use_groundtruth_rel:
            assert c_rel is not None, "Need to provide `c_rel`"
            c_state, c_attn = self.encode_c(c, c_len, c_rel)
        else:
            c_state, c_attn = self.encode_c(c, c_len)
        if use_groundtruth_rel or use_force_mean:
            self.c_attend = _c_attend_last

        # Drop c_state in pretrain stage
        # c_state = [bs, c_emb_dim]

        prior_mu, prior_log_var = self.prior_head(c_state)
        # (above) = [bs, latent_dim]

        # Repeat mu and log_var
        f_one_to_rep: Callable[[Tensor], Tensor] = \
            lambda x: x.repeat(rep_times, *(1,)*x.dim()).flatten(start_dim=0, end_dim=1)
        # [bs, *] -> [rep_times, bs, *] -> [rep_times*bs, *]
        
        f_rep_to_mul: Callable[[Tensor], Tensor] = \
            lambda x: x.unflatten(dim=0, sizes=(rep_times, bs)).movedim(0, 1)
        # [rep_times*bs, *] -> [rep_times, bs, *] -> [bs, rep_times, *]

        if variational:
            z = self.reparametrize(f_one_to_rep(prior_mu), f_one_to_rep(prior_log_var))
        else:
            z = f_one_to_rep(prior_mu)
        # z = [rep_times*bs, latent_dim]

        z = torch.cat((z, f_one_to_rep(c_state)), dim=1)
        # z = [rep_times*bs, latent_dim+c_emb_dim]

        recon, prd, prd_len = self.decode(z, special_idx, x_max_len)

        recon = f_rep_to_mul(recon)
        prd = f_rep_to_mul(prd)
        prd_len = f_rep_to_mul(prd_len)

        # recon = [bs, rep_times, x_max_len, output_dim]
        # c_attn = [bs, c_max_len]
        # prior_mu = [bs, latent_dim]
        # prior_log_var = [bs, latent_dim]
        # prd = [bs, rep_times, x_max_len]
        # prd_len = [bs, rep_times]

        return (recon, c_attn, 
                prior_mu, prior_log_var, 
                prd, prd_len)

    def encode_x(self, x, x_len) -> Tensor:
        # x = [bs, x_max_len]
        x_state: Tensor
        _, x_state = self.x_encoder(x, x_len)
        # x_state = [bs, bid_num*x_enc_hid_dim*x_enc_hid_layers]
        return x_state

    def encode_c(self, c: Tensor, c_len: Tensor, c_rel: Tensor=None) -> Tensor:
        # c = [bs, c_max_len, c_input_dim]
        c_max_len = c.shape[1]
        c_mask = self.create_mask_by_lengths(c_max_len, c_len-1, device=c.device)
        c_mask[:,0] = False
        # c_mask = [bs, c_max_len]

        if self.c_attend:
            if c_rel is None:
                c_attn: Tensor = self.c_energy(c)[:,:,0]
                c_attn = c_attn.masked_fill(~c_mask, -1e10)
                c_attn = torch.softmax(c_attn, dim=-1)
                # c_attn = [bs, c_max_len]
                c_state = torch.sum(c_attn[:,:,None] * c, dim=1)
                # c_state = [bs, c_input_dim]
            else:
                raise ValueError("Conflict between `c_attend` and `c_rel`")
        else:
            if c_rel is None:
                c_attn = None
                c_state = torch.sum(c_mask[:,:,None] * c, dim=1) / torch.sum(c_mask, dim=1)[:,None]
                # c_state = [bs, c_input_dim]
            else:
                c_attn = None
                c_state = torch.sum(c_rel[:,:,None] * c, dim=1)
                # c_state = [bs, c_input_dim]

        c_state = self.fc_c_downsample(c_state)
        # c_state = [bs, c_emb_dim]

        return c_state, c_attn

    def create_mask_by_lengths(self, max_length: int, lengths: Tensor, 
                               device=torch.device('cpu')) -> Tensor:
        """Example:
        >>> max_length = 5
        >>> lengths = torch.tensor([0, 2, 5, 1])
        >>> create_mask_by_lengths(max_length, lengths)
        tensor([[False, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True,  True,  True],
                [ True, False, False, False, False]])
        """
        mask =  torch.arange(max_length)[None, :] < lengths[:, None]
        if mask.device != device:
            mask = mask.to(device)
        return mask

    def decode_tf(self, z: Tensor, trg: Tensor, trg_len: Tensor) -> Tensor:
        """Teacher force learning
        """
        state = self.fc_latent_upsample(z)
        # Use trg_len-1 to remove eos
        output, _ = self.x_decoder(trg, trg_len-1, state)
        return output

    def decode(self, z: Tensor, special_idx: VocabSpecialIndex, 
            trg_max_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Decode sequences from `z` recurrently

        Return:
        ```
        prd: Tensor([1, 2, 3, eos, pad], [2, 3, 4, 2, 3])
        prd_len: Tensor([4, 0])
        output: [batch size, prd len, output dim]
        ```
        """
        state = self.fc_latent_upsample(z)
        batch_size = z.shape[0]
        device = z.device

        output_lst = []
        prd_lst = []
        prd_len = torch.zeros(batch_size, device='cpu', dtype=int)
        _now = torch.full((batch_size, 1), special_idx.sos, device=device)
        _len = torch.ones(batch_size, device='cpu')

        for step in range(trg_max_len - 1):
            output, state = self.x_decoder(_now, _len, state)
            _now = output.argmax(-1)[:, -1:]
            _now[step < prd_len - 1, 0] = special_idx.pad
            prd_len[(_now[:, 0] == special_idx.eos).cpu() & (prd_len == 0)] = step + 1
            prd_lst.append(_now)
            output_lst.append(output)
            if (prd_len > 0).all():
                break
        
        output = torch.cat(output_lst, dim=1)
        prd = torch.cat(prd_lst, dim=-1)

        return output, prd, prd_len

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def kld_loss(self, 
            mu: Tuple[Tensor, Tensor], log_var: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute Kullback-Leibler divergence between 
        distribution 0 and 1. `KL(p:0:recog_mu, q:1:prior)`
        """
        loss = -0.5 * torch.sum(
            1 + (log_var[0] - log_var[1])
            - (log_var[0] - log_var[1]).exp()
            - (mu[0] - mu[1]).pow(2) / log_var[1].exp()
        )
        return loss

    def recons_loss(self, recon: Tensor, trg: Tensor, 
            trg_special_idx: VocabSpecialIndex) -> Tensor:
        """Compute Reconstruction loss. (Ignone `pad` special idx)
        
        Parameters:
        ```
        trg: Tensor([sos, 2, 3, 4, eos])
        ```
        """
        # Use trg[:, 1:] to remove sos
        loss = F.cross_entropy(recon.flatten(end_dim=-2), trg[:,1:].flatten(), 
            ignore_index=trg_special_idx.pad, reduction='sum')
        return loss

    def bow_loss(self, bow_logits: Tensor, trg: Tensor,
            trg_special_idx: VocabSpecialIndex) -> Tensor:
        """Compute bag-of-words loss. (Ignone all special idx)
        
        Parameters:
        ```
        trg: Tensor([sos, 2, 3, 4, eos])
        ```
        """
        # Use trg[:, 1:] to remove sos
        # It seems that 4090 hate torch.prod
        # nvrtc: error: invalid value for --gpu-architecture (-arch)
        mask = trg[:,:,None].ne(
            torch.tensor(astuple(trg_special_idx), device=trg.device)[None,None,:]
            ).all(-1)
        loss = torch.sum(-F.log_softmax(bow_logits, 1).gather(1, trg) * mask)
        return loss

    def rel_loss(self, c_attn: Tensor, c_rel: Tensor, c_len: Tensor) -> Tensor:
        """Compute relevance loss. (Kullback-Leibler divergence between 
        discrete probability distributions)
        """
        # c_attn = [bs, c_max_len]
        # c_rel = [bs, c_max_len]
        c_max_len = c_attn.shape[1]
        c_mask = self.create_mask_by_lengths(c_max_len, c_len-1, device=c_attn.device)
        c_mask[:,0] = False
        
        c_rel = c_rel.to(c_attn.device)

        assert_prob(c_attn*c_mask, name='c_attn')
        assert_prob(c_rel*c_mask, name='c_rel')

        c_attn = c_attn.masked_select(c_mask)
        c_rel = c_rel.masked_select(c_mask)

        loss = torch.sum(c_rel * (c_rel.log() - c_attn.log()))
        return loss
    