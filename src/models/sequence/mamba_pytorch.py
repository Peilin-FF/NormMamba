""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

from opt_einsum import contract as einsum
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention as sdpa
import copy

from local_attention.local_attention import LocalAttention, LocalFlashAttention
from local_attention.pykeops.local_attention import LocalAttention as LocalAttention_pykeops

try:
    import pykeops
    from pykeops.torch import LazyTensor

    pykeops.set_verbose(False)
    has_pykeops = True

except ImportError:
    has_pykeops = False


# from mamba_ssm import Mamba

# import torch
# import torch.nn as nn

"""Simple, minimal implementation of Mamba in one file of PyTorch.

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""


def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)


def selective_scan(u, dt, A, B, C, D, mode='cumsum'):
    dA = torch.einsum('bld,dn->bldn', dt, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
    
    match mode:
        case 'cumsum':
            dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
            x = dB_u * dA_cumsum
            x = x.cumsum(1) / (dA_cumsum + 1e-12)
            y = torch.einsum('bldn,bln->bld', x, C)
        
            return y + u * D
        
        case 'logcumsumexp':
            dB_u_log = complex_log(dB_u)
            
            dA_star = F.pad(dA[:, 1:].cumsum(1), (0, 0, 0, 0, 1, 0))
            x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
            
            y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)
            return y + u * D


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return self.lm_head(x)

    @staticmethod
    def from_pretrained(pretrained_model_name: str, model=None):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        if model is None:
            config_data = load_config_hf(pretrained_model_name)
            model = Mamba(ModelArgs(
                d_model=config_data['d_model'], 
                n_layer=config_data['n_layer'], 
                vocab_size=config_data['vocab_size'], 
            ))
        
        pretrained_dict = load_state_dict_hf(pretrained_model_name)
        model_dict = model.state_dict()
        
        for k, v in pretrained_dict.items():
            k_new = k.replace('backbone.', '')
            if k_new in model_dict and v.size() == model_dict[k_new].size():
                model_dict[k_new] = pretrained_dict[k]
        
        model.load_state_dict(model_dict)
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        
    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        return self.mixer(self.norm(x)) + x
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        
    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        return self.out_proj(y)

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        return selective_scan(x, delta, A, B, C, D, mode=self.args.scan_mode)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


@TransposedModule
class PytorchMambaBlock(SequenceModule):
    """Wrapper for MultiheadAttention using Mamba for efficient attention processing."""
    def __init__(self, d_model, n_heads, dropout=0.0, *args, bias=True, causal=True, rotary=False, **kwargs):
        super().__init__()
        # Sequence model necessary attributes
        self.d_model = d_model
        self.d_output = d_model

        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.num_heads = n_heads
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        if rotary:
            self.rope = RotaryEmbedding(self.d_k)

        # Initialize Mamba block
        mamba_args = ModelArgs(
            d_model=d_model,
            n_layer=1,  # For single Mamba block
            vocab_size=0,  # Not used in this context
            d_state=16,  # Or any other appropriate value
            expand=2,  # Or any other appropriate value
        )
        self.mamba = MambaBlock(mamba_args)

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """
        src: (B, L, D)
        attn_mask: (B, L, L)
        """
        if key_padding_mask is not None:
            raise NotImplementedError("Key padding not implemented for now with Mamba")
        if state is not None:
            raise NotImplementedError("State not implemented for now with Mamba")

        # Apply rotary positional embeddings if used
        if hasattr(self, 'rope'):
            src = self.rope(src)

        # Process the input through the Mamba block
        src = self.mamba(src)

        # Apply dropout
        src = self.dropout(src)

        return src, None  # None to match output signature of original MultiheadAttention

    def step(self, x, state):
        raise NotImplementedError("Not implemented for now with Mamba")


# @TransposedModule
# class MultiheadAttentionFlash1(SequenceModule):
#     """ Simple wrapper for MultiheadAttention using flash implementation of scaled dot product attention"""
#     def __init__(self, d_model, n_heads, dropout=0.0, *args, bias=True, causal=True, rotary=False, **kwargs):
#         "Take in model size and number of heads."
#         super().__init__()
#         # sequence model nessesary attributes
#         self.d_model = d_model
#         print('------------- d_model -------------')
#         print(self.d_model)
#         self.d_output = d_model
#         print('------------- d_output -------------')
#         print(self.d_output)

#         assert d_model % n_heads == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // n_heads
#         print('------------- d_k -------------')
#         print(self.d_k)

#         self.num_heads = n_heads
#         print('------------- num_heads -------------')
#         print(self.num_heads)

#         self.causal = causal
#         print('------------- causal -------------')
#         print(self.causal)
        
#         self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
#         print('------------- linears -------------')
#         print(self.linears)
#         self.attn = sdpa
#         print('------------- attn -------------')
#         print(self.attn)

#         self.dropout_p = dropout
#         print('------------- dropout_p -------------')
#         print(self.dropout_p)

#         if rotary:
#             self.rope = RotaryEmbedding(self.d_k)

#         print('------------- rotary -------------')
#         print(rotary)


#     def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
#         """
#         src: (B, L, D)
#         attn_mask: (B, L, L)
#         """
#         print('------------- key_padding_maks -------------')
#         print(key_padding_mask)
#         print('------------- state -------------')
#         print(state)

#         if key_padding_mask is not None:
#             raise NotImplementedError("key padding Not implemented for now with module MultiHeadedAttentionFlash")
#         if state is not None:
#             raise NotImplementedError("state Not implemented for now with module MultiHeadedAttentionFlash")

#         causal = self.causal if attn_mask is None else False
#         print('------------- causal -------------')
#         print(causal)
#         nbatches = src.size(0)
#         print('------------- nbatches -------------')
#         print(nbatches)


#         # 1) Do all the linear projections in batch from d_model => num_heads x d_k
#         query, key, value = [
#             lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
#             for lin, x in zip(self.linears, (src, src, src))
#         ]  # (B, H, L, Dk)
#         print('------------- query,key,value -------------')
#         print(query.shape, key.shape, value.shape)


#         # 1.5) Add rotary positional embeddings if used
#         if hasattr(self, 'rope'):
#             print('------------- hasattr(self, rope) -------------')
            
#             query = self.rope(query)
#             key = self.rope(key)

#         # 2) Apply attention on all the projected vectors in batch.
#         print('------------- attn_mask -------------')
#         print(attn_mask)
#         x = self.attn(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=causal)  # (B, H, L, Dk)

#         # 3) "Concat" using a view and apply a final linear.
#         x = (
#             x.transpose(1, 2)
#             .contiguous()
#             .view(nbatches, -1, self.num_heads * self.d_k)
#         )  # (B, L, D)
#         print('------------- x.shape -------------')
#         print(x.shape)
#         del query
#         del key
#         del value
#         print('------------- self.linears[-1](x), None -------------')
#         print(self.linears[-1](x).shape, None)
#         return self.linears[-1](x), None  # None to match output signature of MultiheadAttention

#     def step(self, x, state):
#         raise NotImplementedError("Not implemented for now with module MultiHeadedAttentionFlash")