import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional
from torch.nn.utils import parametrize
from attend import Attention
# Corrected import: L2Norm from fla.modules
from fla.modules.l2norm import L2Norm as EfficientL2Norm
from nswig import NGPTLigerSwiGLUMLP
from adams import AdamSRT
# Corrected import: FusedCrossEntropyLoss from fla.modules
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss

@dataclass
class EfficientNGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    attn_alpha_init_val: float = 0.05
    attn_alpha_init_scale: Optional[float] = None
    mlp_alpha_init_val: float = 0.05
    mlp_alpha_init_scale: Optional[float] = None

    sqk_init_val: float = 1.0
    sqk_init_scale: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    ff_expand_factor: float = 4 * 2 / 3
    su_init_val: float = 1.0
    su_init_scale: float = 1.0
    sv_init_val: float = 1.0
    sv_init_scale: float = 1.0

    sz_init_val: float = 1.0
    sz_init_scale: Optional[float] = None
    ignore_index: int = -100

    base_scale: float = field(init=False)
    head_dim: int = field(init=False)
    ff_intermediate_size: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_head
        self.base_scale = 1.0 / math.sqrt(self.n_embd)
        def default(v, d): return v if v is not None else d
        self.attn_alpha_init_scale = default(self.attn_alpha_init_scale, self.base_scale)
        self.mlp_alpha_init_scale  = default(self.mlp_alpha_init_scale,  self.base_scale)
        self.sqk_init_scale        = default(self.sqk_init_scale,  (1.0 / math.sqrt(self.head_dim)))
        self.sz_init_scale         = default(self.sz_init_scale,  self.base_scale)
        assert self.n_embd % self.n_head == 0
        size = int(self.n_embd * self.ff_expand_factor)
        self.ff_intermediate_size = ((size + 127)//128)*128


class Scale(nn.Module):
    def __init__(self, dim, init_val=1.0, init_scale=1.0):
        super().__init__()
        is_scalar = not isinstance(dim, int) or dim <= 1
        actual_init_scale = init_scale if init_scale is not None else 1.0
        if is_scalar:
            self.param = nn.Parameter(torch.tensor(float(actual_init_scale)))
        else:
            self.param = nn.Parameter(torch.ones(dim) * actual_init_scale)
        self.init_val = init_val
        self.init_scale = actual_init_scale

    def forward(self):
        denom = self.init_scale if self.init_scale != 0 else 1e-8
        return self.param * (self.init_val / denom)


class NGPTBlock(nn.Module):
    def __init__(self, cfg: EfficientNGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(
            hidden_size=cfg.n_embd,
            num_heads=cfg.n_head,
            qkv_bias=cfg.bias,
            rope_theta=cfg.rope_theta,
            norm_eps=cfg.norm_eps,
            sqk_init_val=cfg.sqk_init_val,
            sqk_init_scale=cfg.sqk_init_scale,
        )
        mlp_config = type('MLPConfig', (), {
            'n_embd': cfg.n_embd,
            'ff_intermediate_size': cfg.ff_intermediate_size,
            'bias': cfg.bias,
            'su_init_val': cfg.su_init_val,
            'su_init_scale': cfg.su_init_scale,
            'sv_init_val': cfg.sv_init_val,
            'sv_init_scale': cfg.sv_init_scale,
            'norm_eps': cfg.norm_eps
        })()
        self.mlp = NGPTLigerSwiGLUMLP(mlp_config)
        self.alphaA = Scale(cfg.n_embd, init_val=cfg.attn_alpha_init_val, init_scale=cfg.attn_alpha_init_scale)
        self.alphaM = Scale(cfg.n_embd, init_val=cfg.mlp_alpha_init_val, init_scale=cfg.mlp_alpha_init_scale)
        self.norm = EfficientL2Norm(eps=cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def _update(self, x, y, alpha_scaler: Scale):
        alpha = alpha_scaler()
        alpha = torch.abs(alpha)
        updated_x = x + alpha * (y - x)
        normed_updated_x = self.norm(updated_x)
        return self.dropout(normed_updated_x)

    def slerp(self, x, y, alpha_val):
        dot_product = torch.sum(x * y, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        omega = torch.acos(dot_product)
        sin_omega = torch.sin(omega)
        sin_omega = torch.where(sin_omega == 0, torch.ones_like(sin_omega) * 1e-8, sin_omega)
        scale_x = torch.sin((1.0 - alpha_val) * omega) / sin_omega
        scale_y = torch.sin(alpha_val * omega) / sin_omega
        return scale_x * x + scale_y * y

    def forward(self, x, attention_mask=None):
        attn_output, _, _ = self.attention(x, attention_mask=attention_mask, use_cache=False)
        x1 = self._update(x, attn_output, self.alphaA)
        mlp_output = self.mlp(x1)
        mlp_output_normed = self.norm(mlp_output)
        output = self.slerp(x1, mlp_output_normed, alpha_val=0.5)
        return output


class EfficientNGPTModel(nn.Module):
    def __init__(self, cfg: EfficientNGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.embedding_norm = EfficientL2Norm(eps=cfg.norm_eps)
        self.blocks = nn.ModuleList([NGPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = EfficientL2Norm(eps=cfg.norm_eps)
        self.fc_out = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=cfg.bias)
        self.sz = Scale(cfg.vocab_size, init_val=cfg.sz_init_val, init_scale=cfg.sz_init_scale)
        self.loss_fn = FusedCrossEntropyLoss(
            ignore_index=cfg.ignore_index,
            reduction="mean",
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=0.0
        )

    # Corrected signature: Renamed idx to x
    def forward(self, x, targets=None, attention_mask=None):
        x = self.embeddings(x)
        x = self.embedding_norm(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        logits = self.fc_out(x)
        scaled_logits = logits * self.sz()
        loss = None
        if targets is not None:
            loss = self.loss_fn(scaled_logits.view(-1, scaled_logits.size(-1)), targets.view(-1))
        return scaled_logits, loss


def register_l2norm_parametrization(model):
    print("Registering comprehensive L2Norm parametrization...")
    layers_to_parametrize = []
    if hasattr(model, 'embeddings'):
        layers_to_parametrize.append((model.embeddings, "embeddings.weight"))
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attention'):
            layers_to_parametrize.extend([
                (block.attention.q_proj, f"blocks.{i}.attention.q_proj.weight"),
                (block.attention.k_proj, f"blocks.{i}.attention.k_proj.weight"),
                (block.attention.v_proj, f"blocks.{i}.attention.v_proj.weight"),
                (block.attention.o_proj, f"blocks.{i}.attention.o_proj.weight"),
            ])
        if hasattr(block, 'mlp'):
            layers_to_parametrize.extend([
                (block.mlp.gate_proj, f"blocks.{i}.mlp.gate_proj.weight"),
                (block.mlp.up_proj,   f"blocks.{i}.mlp.up_proj.weight"),
                (block.mlp.down_proj, f"blocks.{i}.mlp.down_proj.weight"),
            ])
    if hasattr(model, 'fc_out'):
        layers_to_parametrize.append((model.fc_out, "fc_out.weight"))

    registered_count = 0
    failed_layers = []
    # Removed slice_dim from loop variable unpacking
    for layer, name in layers_to_parametrize:
         if layer is not None and hasattr(layer, 'weight') and layer.weight is not None:
             try:
                 if not parametrize.is_parametrized(layer, 'weight'):
                     norm_module = EfficientL2Norm(eps=model.cfg.norm_eps)
                     # Corrected call: Removed the dim keyword argument
                     parametrize.register_parametrization(layer, 'weight', norm_module)
                     registered_count += 1
             except Exception as e:
                 failed_layers.append(f"{name} (Error: {e})")

    if registered_count > 0: print(f"Parametrizations registered: {registered_count}")
    if failed_layers: print(f"Parametrization failed for: {', '.join(failed_layers)}")
    if registered_count == 0 and not failed_layers: print("Warning: No new parametrizations registered.")