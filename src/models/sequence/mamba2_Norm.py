""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """
from __future__ import annotations
import sys
import json
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import hydra
sys.path.insert(0,"/root/autodl-tmp/MLRA")
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

sys.path.insert(0,"/root/autodl-tmp/MLRA/src/models/sequence")
from opt_einsum import contract as einsum
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention as sdpa
import copy

from local_attention.local_attention import LocalAttention, LocalFlashAttention
from local_attention.pykeops.local_attention import LocalAttention as LocalAttention_pykeops

from mamba_ssm1.Mamba2_Norm import Mamba2_Norm



class RotaryEmbedding(nn.Module):
    """rope = RotaryEmbedding(64)
    q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
    k = torch.randn(1, 8, 1024, 64) # keys

    # apply embeds to queries, keys after the heads have been split out but prior to dot products

    q = rope(q)
    k = rope(k)

    # then do attention with queries (q) and keys (k) as usual

    q_k = [q_k_0,...,q_k_d-1] is mapped to [...,cos(theta**(2mk/d))q_k_2m - sin(theta**(2mk/d))q_k_2m+1, cos(theta**(2mk/d))q_k_2m+1 + sin(theta**(2mk/d))q_k_2m,...]
    """

    def __init__(self, d, theta=10000):
        super().__init__()
        assert d % 2 == 0
        freqs = theta ** (-torch.arange(0, d, 2) / d)  # (d / 2)
        self.register_buffer('freqs', freqs)
        self.cache = dict()

    def get_freqs(self, pos, cache_key=None):
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs  # (d/2)
        freqs = pos.to(freqs).view(-1, 1) * freqs  # (L d/2)

        cos, sin = freqs.cos(), freqs.sin()
        freqs = torch.stack((cos, -sin, sin, cos), dim=-1)  # (L d/2 4)
        freqs = rearrange(freqs, '... (r c) -> ... r c', c=2)  # (L d/2 2 2)

        if cache_key:
            self.cache[cache_key] = freqs

        return freqs  # (L d/2 2 2)

    def forward(self, x, seq_dim=-2):
        # x: (... L d)
        L = x.shape[seq_dim]
        freqs = self.get_freqs(torch.arange(L, device=x.device), L)  # (L d/2 2 2)
        x = rearrange(x, '... (d r) -> ... d r', r=2)  # (... L d/2 2)
        x = einsum('... r c, ... c -> ... r', freqs, x)  # (L d/2 2 2), (... L d/2 2)
        return rearrange(x, '... d r -> ... (d r)')


@TransposedModule
class CudaMambaBlock(SequenceModule):
    """Wrapper for MultiheadAttention using Mamba for efficient attention processing."""
    def __init__(self, 
                 d_model, 
                 dropout=0.0, 
                 ss_state=16, 
                 d_conv=4, 
                 expand=2, 
                 *args, 
                 bias=True, 
                 causal=True, 
                 rotary=False, 
                 **kwargs):

        super().__init__()
        # Sequence model necessary attributes
        self.d_model = d_model
        self.d_output = d_model

        #self.d_k = d_model // n_heads
        #self.num_heads = n_heads
        
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.ss_state = ss_state
        self.d_conv = d_conv
        self.expand = expand

        if rotary:
            self.rope = RotaryEmbedding(self.d_model)

        # Initialize Mamba block
        print(f'-------------------------------')
        print(f'Mamba d_state = {self.ss_state}')
        self.mamba = Mamba2_Norm(d_model = d_model,
                           d_state = self.ss_state,
                           d_conv = self.d_conv,
                           expand = self.expand,
                           headdim = 32,
                           conv_bias = True,
                           bias = False,)

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
