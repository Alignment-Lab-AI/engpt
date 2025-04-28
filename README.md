# nGPT: Normalized Transformer Implementation

This repository contains the PyTorch implementation of the Normalized Transformer (nGPT), an architecture designed for representation learning on the hypersphere, as described in the paper "NGPT: NORMALIZED TRANSFORMER WITH REPRESENTATION LEARNING ON THE HYPERSPHERE" (Published as a conference paper at ICLR 2025).


## Core Concepts

nGPT modifies the standard Transformer by enforcing normalization constraints throughout the network, based on the principle that embeddings, matrix components, and hidden states reside on a unit hypersphere.

1.  **Hypersphere Representation:** All vector representations (token embeddings, rows/columns of weight matrices involved in projections, hidden states) are normalized to unit L2 norm. Matrix-vector multiplications are thus interpreted as sequences of dot products (cosine similarities) bounded within [-1, 1]. (Paper, Sec 1, Sec 2.1, Sec 2.3.2, Sec 2.4.2).
2.  **Normalization Strategy:**
    *   `EfficientL2Norm` (from `fla.modules.l2norm`) is used extensively instead of LayerNorm or RMSNorm. It applies L2 normalization along the feature dimension (`dim=-1`). (Paper, Sec 2.2.2, Table 1).
    *   Weights of linear projections (`nn.Linear`) and embedding tables (`nn.Embedding`) are normalized along their embedding dimension post-initialization and potentially via parametrization during training. (Paper, Sec 2.1, Sec 2.3.2, Sec 2.4.2, Sec 2.6).
    *   Hidden states (`h`) are explicitly normalized after updates. (Paper, Eq 10, 11).
3.  **Learnable Scaling Factors (`Scale` module):** Since normalization removes magnitude information, learnable scaling factors are introduced at various points. The `Scale` module (`engpt.py`, `nswig.py`) implements these factors.
    *   It stores a parameter initialized to `init_scale`.
    *   During the forward pass, it returns `param * (init_val / init_scale)`, effectively allowing the parameter's "functional" value to start at `init_val` while potentially having a different initial learning rate scale controlled by `init_scale`. (Paper, Sec 2.5).
    *   These factors (`sqk`, `su`, `sv`, `sz`, `alphaA`, `alphaM`) control the influence of normalized components.
4.  **Hidden State Update:** The standard residual connection `h = h + sublayer(norm(h))` is replaced with a LERP-like (Linear Interpolation) update on the hypersphere: `h = Norm(h + alpha * (Norm(sublayer_output) - h))`. (Paper, Sec 2.2.2, Eq 10, 11).
    *   `alphaA` and `alphaM` are learnable, per-dimension "eigen learning rates" (implemented via the `Scale` module) controlling the step size towards the normalized attention (`ha`) and MLP (`hM`) outputs, respectively.
    *   The `Norm(...)` acts as a retraction step, projecting the updated vector back onto the unit hypersphere. (Paper, Sec 2.2.2, Appendix A.4).

## Architectural Modifications (vs. Standard Transformer)

Key changes relative to a typical GPT-style decoder architecture:

1.  **Token Embeddings & Output Logits (`engpt.py`):**
    *   Input (`E_input`) and Output (`E_output`) embedding matrices are L2 normalized along the embedding dimension. (Paper, Sec 2.1).
    *   Output logits `z = E_output @ h` (where `h` is the final normalized hidden state) are scaled element-wise by a learnable vector `sz` before the softmax: `z_scaled = z * sz()`. (Paper, Sec 2.1, Eq 1, 3). `sz` is managed by a `Scale` module.
2.  **Attention Block (`attend.py`, `Attention` module):**
    *   Input `h` is *not* pre-normalized (unlike standard `LayerNorm(h)`).
    *   Q, K, V projections (`q_proj`, `k_proj`, `v_proj`): Weights are L2 normalized (via parametrization, see Implementation Details).
    *   RoPE is applied to Q and K as usual.
    *   **Q/K Normalization:** After RoPE, Q and K vectors are explicitly L2 normalized: `q = Norm(q)`, `k = Norm(k)`. (Paper, Sec 2.3.2, Eq 15, 16).
    *   **Q/K Scaling:** Normalized Q and K are scaled by a learnable per-head-dimension vector `sqk`: `q = q * sqk()`, `k = k * sqk()`. `sqk` is managed by a `Scale` module. (Paper, Sec 2.3.2, Eq 15, 16).
    *   **Softmax Scaling:** The scaling factor applied *before* the softmax is `sqrt(head_dim)` instead of the standard `1 / sqrt(head_dim)`. This accounts for the unit variance target after normalizing Q and K. (Paper, Sec 2.3.2).
    *   Flash Attention (`flash_attn_func` or `flash_attn_varlen_func`) is used as the backend. Requires `torch.float16` or `torch.bfloat16`.
    *   Output Projection (`o_proj`): Weights are L2 normalized (via parametrization).
    *   **Output Normalization:** The final output of the attention block (`ha` before the residual update) is L2 normalized: `ha = out_norm(o)`. This `ha` is then used in the hidden state update `h = Norm(h + alphaA * (ha - h))`. (See `attend.py` `out_norm` and `engpt.py` `NGPTBlock`).
3.  **MLP Block (`engpt.py`, `NGPTBlock`, `nswig.py`, `NGPTLigerSwiGLUMLP`):**
    *   Input `h` is *not* pre-normalized.
    *   Uses a SwiGLU variant (`NGPTLigerSwiGLUMLP`) implemented with a custom Triton kernel (`_swiglu_forward_kernel`, `_swiglu_backward_kernel` in `nswig.py`).
    *   Gate (`gate_proj`) and Up (`up_proj`) projection weights are L2 normalized (via parametrization).
    *   **Intermediate Scaling:** The outputs of the up and gate projections (`u`, `v`) are scaled by learnable per-intermediate-dimension vectors `su` and `sv`: `u = u * su()`, `v = v * sv() * sqrt(hidden_size)`. `su` and `sv` are managed by `Scale` modules. (Paper, Sec 2.4.2, Eq 20, 21).
    *   **V Rescaling:** The `v` projection is additionally scaled by `sqrt(hidden_size)` before the SiLU activation within the SwiGLU, intended to place the input into a more favorable regime for the non-linearity. (Paper, Sec 2.4.2, Eq 21, Appendix A.1).
    *   Down Projection (`down_proj`): Weights are L2 normalized (via parametrization).
    *   **Output Normalization:** The final output of the MLP block (`hM` before the residual update) is L2 normalized: `hM = mlp.norm(mlp_output)`. This `hM` is then used in the hidden state update. The paper uses `h = Norm(h + alphaM * (hM - h))` (Eq 11), while the code uses a SLERP between `x1` (post-attention update) and `hM`: `output = self.slerp(x1, hM, alpha_val=0.5)`. The LERP formulation `x1 + alphaM * (hM - x1)` is conceptually similar for small steps. (See `engpt.py` `NGPTBlock`).
4.  **Normalization Layers:** Standard LayerNorm/RMSNorm layers are removed. Normalization is handled by `EfficientL2Norm` applied directly or via parametrization, and explicit `Norm()` calls on hidden states/intermediate outputs. (Paper, Sec 2.6, Point 1).
5.  **Weight Decay:** Not required due to inherent norm control via normalization. Learning rate warmup may also be removed. (Paper, Sec 1, Sec 2.6, Point 7).

## Implementation Details

*   **Modules:**
    *   `attend.Attention`: Implements the nGPT attention mechanism.
    *   `nswig.NGPTLigerSwiGLUMLP`: Implements the custom SwiGLU MLP with Triton backend and scaling factors.
    *   `engpt.Scale`: General module for learnable scalars/vectors with `init_val`/`init_scale` logic.
    *   `fla.modules.l2norm.L2Norm`: Efficient L2 normalization implementation.
    *   `engpt.NGPTBlock`: Combines Attention and MLP blocks with the LERP/SLERP update mechanism and eigen learning rates.
    *   `engpt.EfficientNGPTModel`: Top-level model orchestrating embeddings, blocks, final normalization, and output projection.
*   **Weight Parametrization (`engpt.register_l2norm_parametrization`):**
    *   Uses `torch.nn.utils.parametrize.register_parametrization` to enforce L2 normalization on the `weight` attribute of specified `nn.Linear` layers and `nn.Embedding`.
    *   Applies `EfficientL2Norm` as the parametrization function. This ensures weights have unit norm along the output dimension (for Linear layers) or embedding dimension (for Embeddings) *before* they are used in the forward pass during training.
*   **Dependencies:** Key libraries include `torch`, `einops`, `transformers` (for logging/utils), `triton` (for MLP kernel), `flash-attn`, `fla` (for L2Norm, FusedCrossEntropy).
*   **Data Types:** The implementation, particularly `attend.py`, emphasizes `torch.float16` for compatibility with Flash Attention and potential performance benefits. Explicit `.to(dtype)` calls are used. The Triton kernel in `nswig.py` uses `tl.float32` for internal computations before casting back to the input dtype.
*   **Loss Function:** `fla.modules.fused_cross_entropy.FusedCrossEntropyLoss` is used.
*   **Optimizer:** The code mentions `AdamSRT` (likely a typo for AdamW variant) but configuration specifies Adam/AdamW with zero weight decay. (Paper, Table 3).

## Configuration (`EfficientNGPTConfig`)

The `engpt.EfficientNGPTConfig` dataclass holds hyperparameters. Key nGPT-specific parameters include:

*   `attn_alpha_init_val`, `attn_alpha_init_scale`: Controls the attention eigen learning rate (`alphaA`).
*   `mlp_alpha_init_val`, `mlp_alpha_init_scale`: Controls the MLP eigen learning rate (`alphaM`).
*   `sqk_init_val`, `sqk_init_scale`: Controls the Q/K scaling factor (`sqk`). Default `init_scale` aims for initial `sqrt(dk)` scaling factor if softmax scale is 1, or `d_k^(1/4)` if softmax scale is `sqrt(dk)`. (Paper Sec 2.3.2, footnote)
*   `su_init_val`, `su_init_scale`: Controls the MLP `u` scaling factor (`su`).
*   `sv_init_val`, `sv_init_scale`: Controls the MLP `v` scaling factor (`sv`).
*   `sz_init_val`, `sz_init_scale`: Controls the final logit scaling factor (`sz`).
*   `norm_eps`: Epsilon value for `EfficientL2Norm`.
*   `rope_theta`: Base for Rotary Position Embeddings.

The `__post_init__` method sets default `*_init_scale` values based on embedding/head dimensions if not provided, aiming for sensible initial magnitudes.

