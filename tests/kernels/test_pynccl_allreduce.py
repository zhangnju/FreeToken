"""Correctness test for the RCCL/NCCL-backed pynccl all_reduce (``freetoken.kernel.pynccl``).

Spawns two ranks, runs ``comm.all_reduce(x, "sum")`` on bf16/fp16 tensors and checks the
result bit-for-bit against ``torch.distributed`` as the oracle. On ROCm this exercises the
RCCL port; on CUDA, NCCL. Needs >=2 visible GPUs (skipped otherwise). pynccl only registers
the dtypes the TP engine all-reduces (bf16 / fp16); fp32 is intentionally unsupported.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="pynccl all_reduce needs >=2 GPUs"
)

_CASES = [
    ((1, 2048), torch.bfloat16),
    ((4, 4096), torch.float16),
    ((2, 7, 333), torch.bfloat16),
    ((8192, 2048), torch.bfloat16),
]


def _worker(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    gloo = dist.new_group(backend="gloo")
    dev = torch.device(f"cuda:{rank}")

    from freetoken.kernel.pynccl import init_pynccl

    comm = init_pynccl(tp_rank=rank, tp_size=world, tp_cpu_group=gloo, max_size_bytes=0)
    expected = float(world * (world + 1) // 2)  # rank r contributes (r + 1)
    try:
        for shape, dt in _CASES:
            base = torch.full(shape, float(rank + 1), dtype=dt, device=dev)
            got = base.clone()
            comm.all_reduce(got, "sum")  # the code under test
            ref = base.clone()
            dist.all_reduce(ref)  # torch oracle
            assert torch.equal(got, ref), f"pynccl != torch for {shape} {dt}"
            assert torch.all(got.float() == expected), f"wrong sum for {shape} {dt}"
    finally:
        dist.destroy_process_group()


def test_pynccl_all_reduce_matches_torch() -> None:
    world = 2
    mp.spawn(_worker, args=(world, 29573), nprocs=world, join=True)
