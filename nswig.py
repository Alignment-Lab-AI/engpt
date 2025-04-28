import functools
import importlib
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from packaging.version import Version
from dataclasses import dataclass
from typing import Optional
import math
# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def is_hip() -> bool:
    return torch.version.hip is not None

def calculate_settings(n: int):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"n={n} exceeds max block size {MAX_FUSED_SIZE}")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        args = [x.contiguous() if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)
    return wrapper

def compare_version(package: str, operator_fn: operator, target: str) -> bool:
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    return operator_fn(Version(pkg.__version__), Version(target))

def infer_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def get_amp_custom_fwd_bwd():
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    elif device == 'cuda':
        return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd
    else:
        return (lambda fn: fn), (lambda fn: fn)

amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()

from fla.modules.l2norm import L2Norm as EfficientL2Norm

# --- Scale Module Definition ---
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

# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------

@triton.jit
def silu(x):
    # Cast to float32 before operations
    x_f32 = x.to(tl.float32)
    return (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)

@triton.jit
def _swiglu_forward_kernel(
    gate_ptr, up_ptr, out_ptr, su_ptr, sv_ptr,
    stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, H: tl.constexpr, SQRT_H_FLOAT: tl.float32
):
    pid = tl.program_id(0).to(tl.int64)
    gate_ptr += pid * stride
    up_ptr   += pid * stride
    out_ptr  += pid * stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    
    # Load data
    raw_gate = tl.load(gate_ptr + offs, mask=mask, other=0.)
    raw_up   = tl.load(up_ptr   + offs, mask=mask, other=0.)
    scale_su = tl.load(su_ptr + offs, mask=mask, other=1.0)
    scale_sv = tl.load(sv_ptr + offs, mask=mask, other=1.0)
    
    # Cast to float32 for computations
    raw_gate_f32 = raw_gate.to(tl.float32)
    raw_up_f32 = raw_up.to(tl.float32)
    scale_su_f32 = scale_su.to(tl.float32)
    scale_sv_f32 = scale_sv.to(tl.float32)
    
    gate_scaled = raw_gate_f32 * scale_sv_f32 * SQRT_H_FLOAT
    up_scaled = raw_up_f32 * scale_su_f32
    
    # Sigmoid in float32 precision
    sig_gate = tl.sigmoid(gate_scaled)
    activated = up_scaled * (gate_scaled * sig_gate)
    
    # Store result, convert back to float16 for output
    tl.store(out_ptr + offs, activated.to(raw_gate.dtype), mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    dout_ptr, gate_ptr, up_ptr, su_ptr, sv_ptr,
    dgate_ptr, dup_ptr, dsu_ptr, dsv_ptr,
    stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, H: tl.constexpr, SQRT_H_FLOAT: tl.float32
):
    pid = tl.program_id(0).to(tl.int64)
    dout_ptr   += pid * stride
    gate_ptr   += pid * stride
    up_ptr     += pid * stride
    dgate_ptr  += pid * stride
    dup_ptr    += pid * stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    
    # Load data
    raw_gate = tl.load(gate_ptr + offs, mask=mask, other=0.)
    raw_up   = tl.load(up_ptr   + offs, mask=mask, other=0.)
    scale_su = tl.load(su_ptr + offs, mask=mask, other=1.0)
    scale_sv = tl.load(sv_ptr + offs, mask=mask, other=1.0)
    d_out    = tl.load(dout_ptr + offs, mask=mask, other=0.)
    
    # Cast to float32 for computations
    raw_gate_f32 = raw_gate.to(tl.float32)
    raw_up_f32 = raw_up.to(tl.float32)
    scale_su_f32 = scale_su.to(tl.float32)
    scale_sv_f32 = scale_sv.to(tl.float32)
    d_out_f32 = d_out.to(tl.float32)
    
    gate_scaled = raw_gate_f32 * scale_sv_f32 * SQRT_H_FLOAT
    up_scaled = raw_up_f32 * scale_su_f32
    
    # Sigmoid in float32 precision
    sig_gate = tl.sigmoid(gate_scaled)
    silu_gate = gate_scaled * sig_gate
    silu_deriv = sig_gate * (1.0 + gate_scaled * (1.0 - sig_gate))
    
    d_up_scaled = d_out_f32 * silu_gate
    d_gate_scaled = d_out_f32 * up_scaled * silu_deriv
    d_raw_gate = d_gate_scaled * scale_sv_f32 * SQRT_H_FLOAT
    d_raw_up = d_up_scaled * scale_su_f32
    d_scale_sv = d_gate_scaled * raw_gate_f32 * SQRT_H_FLOAT
    d_scale_su = d_up_scaled * raw_up_f32
    
    # Store results, convert back to original dtype
    tl.store(dgate_ptr + offs, d_raw_gate.to(raw_gate.dtype), mask=mask)
    tl.store(dup_ptr + offs, d_raw_up.to(raw_up.dtype), mask=mask)
    tl.atomic_add(dsu_ptr + offs, d_scale_su.to(scale_su.dtype), mask=mask)
    tl.atomic_add(dsv_ptr + offs, d_scale_sv.to(scale_sv.dtype), mask=mask)

# ----------------------------------------------------------------------
# Python wrappers
# ----------------------------------------------------------------------

def swiglu_forward(a, b, su_scaled, sv_scaled, H, sqrt_H_float): # Takes scaled values
    orig_shape = a.shape
    n_cols = orig_shape[-1]
    a2 = a.view(-1, n_cols)
    b2 = b.view(-1, n_cols)
    out = torch.empty_like(a2)
    n_rows = a2.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    grid = (n_rows,)
    # Pass scaled su/sv directly to kernel
    _swiglu_forward_kernel[grid](
        b2, a2, out, su_scaled, sv_scaled, # Gate=b, Up=a
        b2.stride(0),
        n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE, H=H, SQRT_H_FLOAT=sqrt_H_float, num_warps=num_warps
    )
    return out.view(orig_shape)

def swiglu_backward(dout, saved_a, saved_b, saved_su_scaled, saved_sv_scaled, H, sqrt_H_float): # Takes scaled values
    orig_shape = dout.shape
    n_cols = orig_shape[-1]
    dout2 = dout.view(-1, n_cols)
    db = torch.empty_like(dout2)
    da = torch.empty_like(dout2)
    dsu_proxy = torch.zeros_like(saved_su_scaled)
    dsv_proxy = torch.zeros_like(saved_sv_scaled)
    n_rows = dout2.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    grid = (n_rows,)
    # Pass scaled su/sv directly to kernel
    _swiglu_backward_kernel[grid](
        dout2, saved_b.view(-1, n_cols), saved_a.view(-1, n_cols), saved_su_scaled, saved_sv_scaled,
        db, da, dsu_proxy, dsv_proxy, # dgate=db, dup=da
        dout2.stride(0),
        n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE, H=H, SQRT_H_FLOAT=sqrt_H_float, num_warps=num_warps
    )
    # Return grads: d_gate, d_up, grad for su_scaled, grad for sv_scaled
    return db.view(orig_shape), da.view(orig_shape), dsu_proxy, dsv_proxy

# ----------------------------------------------------------------------
# Autograd Function and nn.Module
# ----------------------------------------------------------------------

class LigerSiLUMulFunctionNGPT(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    @ensure_contiguous
    def forward(ctx, gate_proj_out, up_proj_out, su_param, sv_param, H, sqrt_H_float, su_scaler, sv_scaler):
        su_scaled = su_scaler.forward() # Compute scaled value using Scale module
        sv_scaled = sv_scaler.forward() # Compute scaled value using Scale module
        # Save tensors requiring grad + scaled values + context needed for backward
        ctx.save_for_backward(gate_proj_out, up_proj_out, su_param, sv_param, su_scaled, sv_scaled, torch.tensor(sqrt_H_float))
        ctx.H = H
        ctx.su_scaler = su_scaler # Need scaler object for init_val/init_scale
        ctx.sv_scaler = sv_scaler
        output = swiglu_forward(up_proj_out, gate_proj_out, su_scaled, sv_scaled, H, sqrt_H_float) # Pass scaled values
        return output

    @staticmethod
    @amp_custom_bwd
    @ensure_contiguous
    def backward(ctx, grad_out):
        gate_proj_out, up_proj_out, su_param, sv_param, su_scaled, sv_scaled, sqrt_H_tensor = ctx.saved_tensors
        H = ctx.H
        sqrt_H_float = sqrt_H_tensor.item()
        su_scaler = ctx.su_scaler
        sv_scaler = ctx.sv_scaler
        # Compute grad w.r.t. inputs and *scaled* su/sv
        d_gate, d_up, dsu_proxy, dsv_proxy = swiglu_backward(grad_out, up_proj_out, gate_proj_out, su_scaled, sv_scaled, H, sqrt_H_float)
        # Compute grad w.r.t. underlying parameters using chain rule
        su_denom = su_scaler.init_scale if su_scaler.init_scale != 0 else 1e-8
        sv_denom = sv_scaler.init_scale if sv_scaler.init_scale != 0 else 1e-8
        grad_su_param = dsu_proxy * (su_scaler.init_val / su_denom)
        grad_sv_param = dsv_proxy * (sv_scaler.init_val / sv_denom)
        # Return grads for inputs: gate, up, su_param, sv_param, H, su_scaler, sv_scaler
        return (
            d_gate.to(torch.float32),
            d_up.to(torch.float32),
            grad_su_param.to(torch.float32),
            grad_sv_param.to(torch.float32),
            None, None, None, None
        )


@dataclass
class _NGPTConfig:
    n_embd: int = 768
    ff_intermediate_size: int = 2048
    bias: bool = False
    su_init_val: float = 1.0
    su_init_scale: float = 1.0
    sv_init_val: float = 1.0
    sv_init_scale: float = 1.0
    norm_eps: float = 1e-5

class NGPTLigerSwiGLUMLP(nn.Module):
    def __init__(self, config: _NGPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.sqrt_H = math.sqrt(float(self.hidden_size))
        self.intermediate_size = config.ff_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        # Use Scale modules
        self.su_scaler = Scale(config.ff_intermediate_size, init_val=config.su_init_val, init_scale=config.su_init_scale)
        self.sv_scaler = Scale(config.ff_intermediate_size, init_val=config.sv_init_val, init_scale=config.sv_init_scale)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Pass .param tensors and the scaler modules themselves to apply
        silu_result = LigerSiLUMulFunctionNGPT.apply(
            gate, up,
            self.su_scaler.param, self.sv_scaler.param, # Pass the nn.Parameter tensors
            self.hidden_size,
            self.sqrt_H,
            self.su_scaler, self.sv_scaler # Pass the nn.Module objects
        )
        output = self.down_proj(silu_result)
        return output