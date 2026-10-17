"""Radeon-ops acceleration for gpt-oss-20b decode on RDNA (gfx1100/gfx1201).

Routes two hot paths to the hand-written HIP kernels in the `radeon_ops` library:
  * MXFP4 MoE decode GEMV  -> radeon run_moe_decode_gemv_mxfp4_gptoss (beats the triton split-K kernel)
  * attention qkv/o + lm_head dense bf16 GEMM -> radeon run_dense_gemv_fp8_g32 (per-32-K fp8, lossless)

Call `freetoken.radeon_opt.enable()` once before model construction (idempotent). Requires `radeon_ops`
importable (set RADEON_OPS_PATH if not installed) and RADEON_OPS_LIB pointing at libradeon_ops.so. The dense
path quantizes each weight ONCE (on CPU) and keeps a bf16 dequant for prefill so prefill+decode stay
consistent (decode-only quant drifts); K is padded to a multiple of 128. MoE banks map 1:1 (arg reorder).
"""
from __future__ import annotations

import os
import sys

_applied = False


def _ensure_radeon_on_path():
    p = os.environ.get("RADEON_OPS_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _dense_gemv():
    _ensure_radeon_on_path()
    from radeon_ops.backends.hip.native.moe import run_dense_gemv_fp8_g32
    return run_dense_gemv_fp8_g32


def _dense_bf16():
    _ensure_radeon_on_path()
    from radeon_ops.backends.hip.native.moe import run_dense_gemv_bf16
    return run_dense_gemv_bf16


def _moe_decode():
    _ensure_radeon_on_path()
    from radeon_ops.backends.hip.native.moe import run_moe_decode_gemv_mxfp4_gptoss
    return run_moe_decode_gemv_mxfp4_gptoss


# ---- per-32-K fp8 weight quant (CPU; returns fp8 weight, per-row/per-32 scale, bf16 dequant for prefill) ----
def _quant_g32(weight):
    import torch
    dev = weight.device
    w = weight.detach().to("cpu", torch.float32)
    N, K = w.shape                              # K must be a multiple of 32 (gpt-oss attn/lm_head: 2880/4096)
    wb = w.reshape(N, K // 32, 32)
    scale = wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / 448.0
    qf = (wb / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    wq = qf.reshape(N, K).contiguous().to(dev)
    sinv = scale.reshape(N, K // 32).to(torch.bfloat16).contiguous().to(dev)
    deq = (qf.float() * scale).reshape(N, K).to(torch.bfloat16).contiguous().to(dev)
    return wq, sinv, deq


def _cache(mod):
    if not hasattr(mod, "_radeon_wq"):
        mod._radeon_wq, mod._radeon_sinv, mod._radeon_deq = _quant_g32(mod.weight)
    return mod._radeon_wq, mod._radeon_sinv, mod._radeon_deq


def _patch_dense():
    import torch.nn.functional as F
    from freetoken.core import get_global_ctx
    from freetoken.layers.embedding import ParallelLMHead
    from freetoken.layers.linear import _LinearTPImpl, LinearOProj

    def _decode_m1():
        b = get_global_ctx().batch
        return b.size == 1 and not b.is_prefill

    orig_lm = ParallelLMHead.forward

    def lm_forward(self, x):
        try:
            if self.tied_embedding is None and getattr(self, "tp_size", 1) == 1 and hasattr(self, "weight"):
                wq, sinv, _ = _cache(self)   # pre-quantize at prefill too (before decode-graph capture)
                if _decode_m1():
                    return _dense_gemv()(x.reshape(1, -1), wq, sinv)[:, : self.num_embeddings]
        except Exception as e:  # pragma: no cover
            print(f"[radeon_opt] lm_head fallback: {e}", file=sys.stderr, flush=True)
        return orig_lm(self, x)

    ParallelLMHead.forward = lm_forward

    orig_lin = _LinearTPImpl.forward
    orig_o = LinearOProj.forward

    def dense_forward(self, x, orig):
        try:
            if getattr(self, "_tp_size", 1) in (0, 1) and hasattr(self, "weight") \
                    and x.shape[-1] == self.weight.shape[1]:
                bias = getattr(self, "bias", None)
                b = get_global_ctx().batch
                # gpt-oss MoE router gate (uniquely tiny N=num_experts): EXACT bf16 GEMV -- it feeds top-k, so
                # no quant; the win is removing the per-call rocBLAS tiny-GEMM that serialises the decode graph.
                if self.weight.shape[0] <= 64:
                    if b.is_prefill or b.size > 1:
                        return orig(self, x)
                    y = _dense_bf16()(x.reshape(1, -1), self.weight)[:, : self.weight.shape[0]]
                    return y + bias if bias is not None else y
                # attention qkv/o projections: per-32 fp8 GEMV (lossless)
                if getattr(self, "prefix", "").endswith(("qkv_proj", "o_proj")):
                    wq, sinv, deq = _cache(self)
                    if b.is_prefill or b.size > 1:
                        return F.linear(x, deq, bias)   # prefill: dequantized fp8 weights (consistent)
                    y = _dense_gemv()(x.reshape(1, -1), wq, sinv)[:, : self.weight.shape[0]]
                    return y + bias if bias is not None else y
        except Exception as e:  # pragma: no cover
            print(f"[radeon_opt] dense fallback: {e}", file=sys.stderr, flush=True)
        return orig(self, x)

    _LinearTPImpl.forward = lambda self, x: dense_forward(self, x, orig_lin)
    LinearOProj.forward = lambda self, x: dense_forward(self, x, orig_o)


def _patch_moe():
    import freetoken.moe.fused_mxfp4 as m
    orig = m.run_mxfp4_splitk_decode_experts

    def wrapper(x, topk_weights, topk_ids, gate_up, gate_up_scale, gate_up_bias,
                down, down_scale, down_bias, top_k, hidden_act_alpha, swiglu_limit):
        try:
            return _moe_decode()(x, gate_up, gate_up_scale, gate_up_bias, down, down_scale, down_bias,
                                 topk_ids, topk_weights, hidden_act_alpha, swiglu_limit)
        except Exception as e:  # pragma: no cover
            print(f"[radeon_opt] moe fallback: {e}", file=sys.stderr, flush=True)
            return orig(x, topk_weights, topk_ids, gate_up, gate_up_scale, gate_up_bias,
                        down, down_scale, down_bias, top_k=top_k,
                        hidden_act_alpha=hidden_act_alpha, swiglu_limit=swiglu_limit)

    m.run_mxfp4_splitk_decode_experts = wrapper


def enable():
    """Idempotently route MXFP4 MoE decode + attn/lm_head dense to radeon_ops kernels."""
    global _applied
    if _applied:
        return
    _patch_moe()
    _patch_dense()
    _applied = True
    print("[radeon_opt] enabled: MXFP4 MoE + per-32 fp8 dense (attn/lm_head)", file=sys.stderr, flush=True)
