"""Block-FP8 routed-expert MoE dispatch (offload `_expert_gemm` "fp8_block" branch).

The experts live in the offload cache as block-fp8 (fp8-e4m3 weight + bf16 per-128x128
``weight_scale_inv``) -- half the resident/host bytes of bf16, so the cache holds ~2x more
experts. The grouped GEMMs read the routed experts' fp8 rows directly and dequantize inside
the K-loop (``kernel/triton/fp8_blockscale_moe``), so no gather/copy or separate bf16 dequant
of the experts is ever materialized. Decode shapes are static per captured batch size, so the
path stays CUDA-graph capturable. Same entry points are reused by the resident (non-offload)
``Fp8ResidentMoE`` -- ``topk_ids`` index expert rows in the resident case and cache slots in
the offload case; both index the bank tensors identically.
"""

from __future__ import annotations


def fused_experts_fp8_block(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, num_experts, activation="silu",
    apply_router_weight_on_input=False, act_alpha=1.0, act_limit=float("inf"),
):
    """Prefill: W8A8 fused grouped GEMM over the materialized-layer banks
    (``[num_experts, ...]``, position == expert id)."""
    assert not apply_router_weight_on_input

    # RDNA native fp8-WMMA grouped prefill GEMM (radeon_ops): W8A8 on gfx1201 fp8 tensor cores
    # (~1.66x bf16-WMMA), vs the emulated-bf16 Triton. Plain-SwiGLU only; weights + activation fp8.
    import torch

    if (
        activation == "silu"
        and act_alpha == 1.0
        and act_limit == float("inf")
        and hidden_states.dtype is torch.bfloat16
        and gate_up.dtype is torch.float8_e4m3fn
    ):
        import os
        from freetoken.kernel.backend import is_radeon_installed

        if is_radeon_installed() and os.environ.get("RADEON_MOE", "1") != "0":
            return _radeon_fp8_prefill(
                hidden_states, gate_up, gate_up_scale, down, down_scale,
                topk_weights, topk_ids, num_experts, activation, act_alpha, act_limit,
            )

    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_fp8_blockscale

    return fused_experts_fp8_blockscale(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, num_experts, activation, act_alpha, act_limit,
    )


def _radeon_fp8_prefill(hidden_states, gate_up, gate_up_scale, down, down_scale,
                        topk_weights, topk_ids, num_experts, activation, act_alpha, act_limit):
    """Native fp8-WMMA prefill: mirrors fused_experts_fp8_blockscale but the two grouped GEMMs run on
    radeon_ops' fp8 tensor-core kernel (moe_align at block_size=16 to match its 16x16 tiles)."""
    import torch

    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.fp8_block_linear import per_token_group_quant_fp8
    from freetoken.layers import gated_act_and_mul
    from freetoken.moe.fused import moe_align_block_size
    from radeon_ops.backends.hip.native.moe import run_moe_prefill_gemm_fp8

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, 16, num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    a1_fp8, a1_scale = per_token_group_quant_fp8(hidden_states, 128)
    ic1 = torch.zeros((M, top_k, two_i), device=dev, dtype=dt)
    run_moe_prefill_gemm_fp8(a1_fp8, a1_scale, gate_up, gate_up_scale, ic1, tw,
                             sorted_ids, expert_ids, ntpp, num_valid, top_k, 0)
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    a2_fp8, a2_scale = per_token_group_quant_fp8(ic2, 128)
    ic3 = torch.zeros((M, top_k, H), device=dev, dtype=dt)
    run_moe_prefill_gemm_fp8(a2_fp8, a2_scale, down, down_scale, ic3, tw,
                             sorted_ids, expert_ids, ntpp, num_valid, 1, 1)
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_fp8_block(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, activation="silu", apply_router_weight_on_input=False,
    act_alpha=1.0, act_limit=float("inf"),
):
    """Decode: W8A16 fused inline-dequant grouped GEMV -- reads the routed experts' fp8 rows
    directly (``topk_ids`` index the banks) and dequantizes in the K-loop. CUDA-graph safe."""
    assert not apply_router_weight_on_input

    # RDNA native fp8 block-scale decode GEMV (radeon_ops): memory-bound W8A16 SwiGLU expert GEMV,
    # ~1.6-3.5x the Triton decode GEMV at T=1 on gfx1201 (wave-per-output vs triton's starved grid).
    # Plain-SwiGLU only; weights fp8-e4m3, activation bf16; returns TP-local [M, H].
    import torch

    if (
        activation == "silu"
        and act_alpha == 1.0
        and act_limit == float("inf")
        and hidden_states.dtype is torch.bfloat16
        and gate_up.dtype is torch.float8_e4m3fn
    ):
        import os
        from freetoken.kernel.backend import is_radeon_installed

        if is_radeon_installed() and os.environ.get("RADEON_MOE", "1") != "0":
            from radeon_ops.backends.hip.native.moe import run_moe_decode_gemv_fp8

            return run_moe_decode_gemv_fp8(
                hidden_states, gate_up, gate_up_scale, down, down_scale, topk_ids, topk_weights
            )

    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_decode_fp8_blockscale

    return fused_experts_decode_fp8_blockscale(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, activation, act_alpha, act_limit,
    )


__all__ = ["fused_experts_fp8_block", "fused_experts_decode_fp8_block"]
