# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modifications for nGPT compatibility marked with # nGPT MOD

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple
import math # nGPT MOD: Need math for sqrt

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
# nGPT MOD: Import EfficientL2Norm instead of RMSNorm
# from fla.modules import RMSNorm, RotaryEmbedding
from fla.modules import RotaryEmbedding
from fla.modules.l2norm import L2Norm as EfficientL2Norm # nGPT MOD

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)

# nGPT MOD: Add the Scale class definition (copied from your code)
class Scale(nn.Module):
    """learnable α or s vectors with init val/scale logic"""
    def __init__(self, dim, init_val=1.0, init_scale=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.ones(dim) * init_scale)
        self.init_val = init_val
        self.init_scale = init_scale # The initial value stored in the parameter

    def forward(self):
        # Returns the properly scaled parameter for use in the forward pass
        return self.param * (self.init_val / self.init_scale)

class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        # qk_norm: bool = False, # nGPT MOD: Remove qk_norm flag, replace with L2Norm
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        # nGPT MOD: Add nGPT specific parameters
        norm_eps: float = 1e-5,
        sqk_init_val: float = 1.0,
        sqk_init_scale: Optional[float] = None, # Default handled in Scale class if None
        dtype: torch.dtype = torch.float16  # Added explicit dtype parameter with fp16 default
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        # self.qk_norm = qk_norm # nGPT MOD: Removed

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.dtype = dtype  # Store dtype for use in forward pass

        # Initialize linear projections with the specified dtype
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias).to(dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias).to(dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias).to(dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype)
        
        self.q_norm = EfficientL2Norm(eps=norm_eps)  # Correct - Only eps
        self.k_norm = EfficientL2Norm(eps=norm_eps)  # Correct - Only eps
        _sqk_init_scale_val = sqk_init_scale if sqk_init_scale is not None else (1.0 / math.sqrt(self.head_dim))
        self.sqk = Scale(self.head_dim, init_val=sqk_init_val, init_scale=_sqk_init_scale_val)
        self.out_norm = EfficientL2Norm(eps=norm_eps)  # Correct - Only eps

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2

        # Ensure input is in the right dtype
        hidden_states = hidden_states.to(self.dtype)
        
        batch_size, q_len, _ = hidden_states.size()

        # Linear projections with explicit casting
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim).to(self.dtype)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim).to(self.dtype)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim).to(self.dtype)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        seqlen_offset, max_seqlen = 0, q_len
        
        # Handle past_key_values and seqlen_offset calculation
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        # Apply RoPE and ensure output is in fp16
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        q, k = q.to(self.dtype), k.to(self.dtype)

        # nGPT MOD: Apply L2Norm and sqk scaling AFTER RoPE
        q = self.q_norm(q).to(self.dtype)
        k = self.k_norm(k).to(self.dtype)
        sqk_scale = self.sqk().to(self.dtype)
        q = (q * sqk_scale).to(self.dtype)
        k = (k * sqk_scale).to(self.dtype)

        # Handle KV Caching
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim).to(self.dtype)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim).to(self.dtype)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention")

        # Ensure all inputs to flash_attn are in fp16
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        # nGPT MOD: Define softmax scale
        softmax_scale = math.sqrt(self.head_dim)

        # Handle padding and call flash_attn (with fp16 inputs)
        if attention_mask is not None:
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            # Ensure tensors are in fp16 after unpadding
            q, k, v = q.to(self.dtype), k.to(self.dtype), v.to(self.dtype)
            
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                softmax_scale=softmax_scale # nGPT MOD
            )
            o = pad_input(o, indices_q, batch_size, q_len).to(self.dtype)
        elif cu_seqlens is not None:
            q_squeezed = q.squeeze(0).to(self.dtype)
            k_squeezed = k.squeeze(0).to(self.dtype)
            v_squeezed = v.squeeze(0).to(self.dtype)
            
            o = flash_attn_varlen_func(
                q_squeezed, k_squeezed, v_squeezed,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                softmax_scale=softmax_scale # nGPT MOD
            ).unsqueeze(0).to(self.dtype)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                softmax_scale=softmax_scale # nGPT MOD
            ).to(self.dtype)

        # Reshape and output projection
        o = o.reshape(batch_size, q_len, -1).to(self.dtype)
        o = self.o_proj(o).to(self.dtype)

        # nGPT MOD: Apply output normalization
        o = self.out_norm(o).to(self.dtype)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values