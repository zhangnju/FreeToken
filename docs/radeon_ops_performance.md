# Radeon 算子库在 FreeToken 上的性能（RDNA）

本文总结 FreeToken 在 AMD RDNA GPU 上使用的 Radeon 算子库 HIP kernel（`radeon_ops`）相对
FreeToken 自带 Triton 算子的实测性能：每个 HIP kernel 的算子级数据，以及接入服务路径后端到端（E2E）
的模型吞吐提升。所有数字均为设备上同输入的 A/B 实测。

## 环境

- **GPU**：AMD Radeon PRO W7900（gfx1100 / RDNA3）、AMD Radeon AI PRO R9700（gfx1201 / RDNA4）。
- **模型**：Qwen3.6-35B-A3B（256 专家，top-8，hidden=2048），bf16 / fp8-block / NVFP4 三种权重格式。
- **接入方式**：Radeon 算子库 HIP kernel 通过环境开关（`RADEON_MOE`、`RADEON_DENSE_FP8`）选用；关闭
  开关即回退 Triton。每组对比都是同一次构建的 A/B。
- **TP1/TP2/TP4**：指每个张量并行 rank 实际计算的本地 intermediate 大小 I = 512 / 256 / 128。

---

## 1. 算子级：Radeon 算子库 HIP kernel vs Triton（实测数值）

### 1.1 MoE 解码 GEMV（B=1，访存受限）

单位 µs/次。加速比 = Triton 时间 / Radeon 算子库 HIP kernel 时间。

**fp8-block 解码 GEMV**（对比 Triton fp8-blockscale）

| 形状 | gfx1100 Triton | gfx1100 HIP kernel | 加速 | gfx1201 Triton | gfx1201 HIP kernel | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| I=512 (TP1) | 67.7 | 56.3 | 1.20x | 71.9 | 48.9 | 1.47x |
| I=256 (TP2) | 66.3 | 39.1 | 1.70x | 72.9 | 33.4 | 2.18x |
| I=128 (TP4) | 65.9 | 27.3 | 2.42x | 64.1 | 27.1 | 2.36x |

**NVFP4 解码 GEMV**（对比 Triton-Marlin）

| 形状 | gfx1100 Marlin | gfx1100 HIP kernel | 加速 | gfx1201 Marlin | gfx1201 HIP kernel | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| I=512 (TP1) | 73.7 | 40.9 | 1.79x | 74.3 | 45.8 | 1.59x |
| I=256 (TP2) | 74.0 | 28.6 | 2.59x | 72.9 | 34.6 | 2.09x |
| I=128 (TP4) | 74.5 | 24.8 | 2.95x | 73.4 | 27.6 | 2.62x |

**bf16 解码 GEMV**（gfx1201，对比 Triton；Triton 因并行度不足基本持平在 ~110µs）

| 形状 | Triton | HIP kernel | 加速 |
| --- | --- | --- | --- |
| I=512 (TP1) | ~110 | ~60 | 1.83x |
| I=256 (TP2) | ~110 | ~37 | 3.00x |
| I=128 (TP4) | ~110 | ~30 | 3.69x |

关键观察：
- Triton 分组核在 B=1 且本地 I 较小时并行度不足，时间基本持平（上表 fp8/NVFP4 的 Triton 列在三个
  形状上几乎不变，bf16 的 Triton 恒定 ~110µs）；HIP kernel 的 wave-per-output 核贴近显存带宽上限，
  时间随 I 下降，所以 TP 越大加速越明显。
- fp8/NVFP4 核用向量化 `uint`/`float2` 读取、group scale 每块只解一次（而非逐元素），把吞吐大致
  翻倍，连 TP1 也超过 1.0x。
- 精度：NVFP4 解码对 torch e2m1 反量化参考相对误差 ~1.4e-3；fp8 解码对 Triton 块量化核 ~3e-3。

### 1.2 MoE prefill 分组 GEMM（M=512）

**NVFP4 tiled prefill（bf16-WMMA）**，gfx1201，完整融合路径（2 个分组 GEMM + silu + sum-reduce）：

| 形状 | Triton | HIP kernel | 加速 | 相对误差 |
| --- | --- | --- | --- | --- |
| inter=256 (TP2) | 2.25 ms | 1.24 ms | 1.82x | 4.6e-7 |
| inter=128 (TP4) | 1.25 ms | 0.79 ms | 1.58x | 4.6e-7 |

（纯 GEMM、动态 tile 调优后的最优可到 1.87x / 1.60x；因两边都用 bf16 计算，结果近乎 bit-exact。）

### 1.3 dense fp8 GEMV（split-K，用于 GDN out_proj 等稠密投影）

gfx1100，单位 µs/次，对比 Triton `block_fp8_linear`：

| 形状 | Triton | HIP kernel | 加速 |
| --- | --- | --- | --- |
| out_proj：K=4096, N=2048 | 25.3 | 12.5 | 2.02x |
| K=2048, N=2048 | 25.4 | 9.9 | 2.56x |
| in_proj：K=2048, N=8192 | 34.3 | 20.5 | 1.67x |
| K=2048, N=16384 | 76.7 | 39.3 | 1.95x |

（bit-exact。split-K 让大 K 的 out_proj 也能赢 Triton；按中等输出规模 gate——大 N 时 Triton 的大 N
GEMV 更优，交给 Triton。gfx1201 的 out_proj 同样 ~2.0x：Triton 28.9µs / HIP kernel 14.4µs。）

### 1.4 GDN（Gated DeltaNet 线性注意力）

**Prefill**（chunked WY 流水线，per GDN layer，GQA 16 K-head / 32 V-head，head_dim 128；对比
FreeToken 当前走的 fla-Triton `chunk_gated_delta_rule`，单位 ms）：

| seqlen | gfx1100 Triton | gfx1100 HIP kernel | 加速 | gfx1201 Triton | gfx1201 HIP kernel | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.639 | 0.384 | 1.67x | 0.567 | 0.438 | 1.30x |
| 2048 | 2.441 | 1.552 | 1.57x | 1.521 | 1.518 | 1.00x |
| 8192 | 10.687 | 6.490 | 1.65x | 5.573 | 5.949 | 0.94x |

精度 max|Δ| 6.1e-5。关键分架构：gfx1100（RDNA3）的 fla-Triton LA 核较弱，HIP kernel 稳定赢
**1.57–1.67x**；gfx1201（RDNA4）的 Triton LA 核快了约 2x（8192: 5.57ms vs gfx1100 10.69ms），
把 HIP kernel 的优势抹平——短序列仍小胜 1.30x，中/长序列持平到 −6%。所以 GDN prefill 的收益集中在
RDNA3。（gfx1201 长序列已用 autotune 最优配置，四配置菜单扫完仍落后，是核调度上限而非调优缺口。）

**Decode**（recurrent step；bf16 state vs fp32 state，均为 HIP kernel——bf16 把 `[B,H,DK,DV]` 状态的
HBM 流量减半。单位 Ktok/s，H=32）：

| batch | gfx1100 fp32-state | gfx1100 bf16-state | 加速 | gfx1201 fp32-state | gfx1201 bf16-state | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| B=1 | 42.5 | 41.8 | 1.0x | 22.5 | 22.6 | 1.0x |
| B=16 | 514.8 | 620.8 | 1.21x | 369.8 | 367.2 | 1.0x |
| B=64 | 146.5 | 740.4 | 5.05x | 111.6 | 611.5 | 5.48x |

B=1 单流解码状态很小，bf16 无收益；batch 增大后 fp32 状态溢出末级缓存，bf16 把流量减半，B=64 达 ~5x。
精度：bf16 对快衰减 head 与 fp32 等同，慢衰减 head 漂移 ~3e-4（caller 可按精度需求选 fp32）。

### 1.5 Attention（Flash-Attention）

**Prefill**（causal GQA，Hq=32 / Hkv=8，head_dim 128；对比 FreeToken 部署的 `extend_paged_attention`
triton 核，prefix_len=0 纯 prefill，单位 ms）：

| seqlen | gfx1100 ft-triton | gfx1100 HIP kernel | 加速 | gfx1201 ft-triton | gfx1201 HIP kernel | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.097 | 0.087 | 1.11x | 0.065 | 0.042 | 1.52x |
| 2048 | 0.870 | 0.820 | 1.06x | 0.480 | 0.424 | 1.13x |
| 8192 | 13.29 | 11.65 | 1.14x | 6.95 | 6.06 | 1.15x |

两架构 prefill 都稳定小胜 FreeToken 的 extend 核 **1.06–1.52x**（gfx1201 短序列最大）。

**Decode（paged）不是赢点**：FreeToken 的 `decode_paged_attention`（SGLang 风格 split-k flash-decoding）
调校充分，native 2D-blocked paged-decode 核在两架构的大 context / 大 batch 都落后（gfx1201 0.59–1.26x、
gfx1100 0.70–1.20x），仅在小 batch × 短 context 偶有小胜。所以 attention 的收益集中在 **prefill**；
paged decode 继续走 FreeToken 的核。

---

## 2. 端到端模型性能（服务）

算子级加速会被 Amdahl 定律摊薄：MoE GEMV 只是解码一步的一部分。HIP kernel 在它是主导成本（bf16
权重大）或在 prefill（计算受限、MoE 占比高）时，E2E 收益最明显。

### 2.1 解码吞吐（tok/s）

| 模型 / 配置 | Triton | HIP kernel | 提升 |
| --- | --- | --- | --- |
| bf16 35B-A3B，TP=4 常驻（gfx1201） | 64.5 | 77.3 | **+20%** |
| fp8 35B-A3B-FP8 + dense fp8 GEMV，TP=1（gfx1100） | 39.5 | 41.3 | **+4.6%** |

- bf16 解码受益最大：权重更大，MoE GEMV 成为瓶颈。
- fp8 把 dense fp8 GEMV HIP kernel 接进大量中等 N 的投影（GDN out_proj 等）后，E2E 解码拿到 +4.6%。

### 2.2 Prefill 首 token 时延（TTFT，1100-token prompt，TP=1，gfx1100）

| 模型 | Triton TTFT | HIP kernel TTFT | 加速 |
| --- | --- | --- | --- |
| NVFP4 35B-A3B-NVFP4 | 210.7 ms | 77.6 ms | **2.72x** |
| fp8 35B-A3B-FP8 | 164.1 ms | 113.7 ms | **1.44x** |

Prefill 是 E2E 最大的赢点：TP=1 下无通信摊薄、长 prompt 里 MoE GEMM 占比大，且 Triton 的 NVFP4
prefill 在 RDNA 上本就弱。

### 2.3 NVFP4 多卡：HIP kernel 是"使能者"

| 模型 / 配置 | Triton | HIP kernel | 结论 |
| --- | --- | --- | --- |
| NVFP4 35B-A3B-NVFP4，TP=2（gfx1201） | 无可用核（`KernelSelectionError`） | **95.4 tok/s** | HIP kernel 是唯一可服务路径 |

RDNA 上 stock FreeToken 没有任何可用的 NVFP4 MoE 核：Triton 核拒绝 TP>1，Marlin/b12x 仅限 CUDA，
启动即 `KernelSelectionError`。所以这里 HIP kernel 不只是"更快"，而是多卡服务 NVFP4 **从不能跑到能跑**。

---

## 3. 结论

- 三种格式（bf16/fp8/NVFP4）的 Radeon 算子库 HIP kernel 解码 GEMV 在所有 TP 形状上都赢 Triton，
  算子级 **1.2x–3.7x**，精度在量化粒度内。
- E2E 解码遵循 Amdahl：bf16 大赢（**+20%**），单独的 fp8 MoE 打平，fp8 再换掉 dense 投影后拿回
  **+4.6%**。
- Prefill 是 E2E 最大赢点（TTFT 快 **1.4x–2.7x**）；NVFP4 多卡服务则完全依赖 Radeon 算子库 HIP
  kernel 才成为可能。
- GDN：prefill 在 RDNA3（gfx1100）稳定赢 fla-Triton **1.57–1.67x**，RDNA4（gfx1201）因 Triton LA
  核更强而持平；decode 的 bf16 state 在高并发批量省状态带宽。
- Attention：prefill 两架构都小胜 FreeToken 的 `extend` 核 **1.06–1.52x**；paged decode 不是赢点
  （FreeToken split-k 已调校充分），继续走 FreeToken 的核。这些收益与架构强相关，需分架构看。
