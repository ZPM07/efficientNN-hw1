# Homework 1 — Analytical performance model of a small CNN

ITMO «Efficient Models» 2026 · HW1 · 5 points

Author: **Смирнов Василий Артурович** (Vasily Arturovich Smirnov)
GPU used for all measurements: **NVIDIA GeForce GTX 1650 (TU117, 4 GB GDDR5, Turing, no tensor cores)**, driver 572.16 (CUDA-capable 12.8; the PyTorch build is cu126, so `torch.version.cuda` reports 12.6).

## Repository layout

```
hw1/
├── README.md                # this file
├── hw1_handwritten.pdf      # scanned hand-written Part 1 derivations
├── models.py                # the assigned SmallCNN
├── equations.py             # flops(), memory(), latency(), energy()  [+ bytes_moved()]
├── measure.py               # Part 3 measurement protocol (grid, latency, memory, energy, OOM, kernels, --vram-cap)
├── calibrate.py             # Part 4 fitting of theta (latency + energy) -> results/theta.json
├── make_figures.py          # Part 5 plots -> results/figures/*.png
├── pyproject.toml           # uv project (torch 2.6.0+cu126, numpy, pandas, scipy, matplotlib, nvidia-ml-py)
└── results/
    ├── measurements.csv     # S, B, latency_s, memory_bytes, energy_J, is_validation ("OOM" rows supported)
    ├── kernels.csv          # S, B, layer (record_function tag), CUDA kernel name (n/a on Windows: no CUPTI, see t_launch note)
    ├── theta.json           # fitted parameters + quality metrics (+ suspected-spillover breakdown)
    ├── measure_log.txt      # console log of the full run (UTF-16)
    └── figures/*.png
```

## How to reproduce

```bash
uv sync                                   # creates .venv, installs torch 2.6.0+cu126 etc.
uv run python measure.py                  # ~30-40 min on a GTX 1650: full 132-config grid
uv run python calibrate.py                # fits theta on train rows, validates on the rest
uv run python make_figures.py             # all plots + parity/quality summary
uv run python equations.py                # self-check of the closed-form formulas
```

`measure.py --quick` runs a 9-config smoke test (~1 min).
`measure.py --vram-cap 0.9` is an **artificial software-limit experiment**: the PyTorch caching allocator is capped at 90 % of VRAM, so allocations beyond the cap raise `torch.cuda.OutOfMemoryError`. An OOM row then means "above the cap", not necessarily "physical VRAM exhausted" (see the WDDM section).

**Environment of the measurement run:** Windows 10 Pro 22H2 (build 19045), Python 3.12.4, PyTorch 2.6.0+cu126 (`torch.version.cuda` = 12.6), NVIDIA driver 572.16, GTX 1650 4 GiB GDDR5.

**Grid reproducibility:** `measure.py` builds the grid with `np.random.default_rng(2026)`: base S = {32, 64, 128, 224, 256, 384, 512} plus 4 random multiples of 16 → **S = {48, 112, 352, 400}**; base B = {1, 2, 4, 8, 16, 32, 64, 128, 256} plus 3 random non-powers-of-two → **B = {95, 98, 167}**; full 11 × 12 = 132 cross product. Split rule: `is_validation = 1` for every row whose S is one of the four random sizes **or** whose B is one of the three random batches → **63 train / 69 validation rows**; `calibrate.py` fits only on the train rows.

## Model of the network (all conventions from the statement)

- FLOPs: 1 MAC = 2 FLOPs; BN(eval) = 2 FLOPs/elem; ReLU = 1/elem; MaxPool = 0 FP-FLOPs (comparisons only: 8 compares × 2S² output elements/sample = 16·B·S² compares); GAP = (n−1 adds + 1 divide) per channel = 2S².
- **FLOPs(S,B) = B·(17 777·S² + 313 600)** — closed form, no calibration; exact *under the stated counting convention* (1 MAC = 2 FLOPs, BN = 2/elem, ReLU = 1/elem, pool = 0), which is not a claim about the number of GPU instructions actually executed (cudnn may decompose a conv into im2col + GEMM, Winograd, etc.). Check: FLOPs(224,1) = 892 292 352; `equations.py` re-counts it by an independent layer-by-layer brute force.
- **Memory(S,B) = 4·(1 045 316 + 19·B·S²)** bytes — ideal refcounting allocator *under the measurement convention*: the input tensor is alive during the whole forward (the caller holds a reference), so the peak stage is the first BatchNorm — network input 3S² + BN input 8S² + BN output 8S² = 19S² floats per sample. GAP output and logits appear later and never coexist with this peak, so they are not added. cuDNN workspaces are allocated through the PyTorch caching allocator (they *do* count in `max_memory_allocated`), so the equation is a *systematic lower bound*.
- **Bytes moved** = Σ(read input + write output) + weights read once = `4·(133·B·S² + 2 148·B + 1 045 316)` — an estimate of the **logical traffic** (every tensor touched once), not literal DRAM traffic: weights/activations can stay in cache across ops and iterations, and a conv may re-read its input.
- **Latency** (calibrated): `T = Σ_ops max(flops_op/P, bytes_op/BW) + 23·t_launch` — a per-op roofline. P, BW, t_launch are **effective** parameters (see below); N = 23 launches is an assumption (one kernel per high-level op), and only the product N·t_launch ≈ 1.82 ms is identified.
- **Energy** (calibrated): `E = c0 + c_f·FLOPs + c_b·Bytes` [J] — an affine regression; the coefficients are **effective** (FLOPs and Bytes are collinear, r = 1.000), not physical per-flop/per-byte costs.

### Measurement protocol
`torch.backends.cudnn.benchmark=False`, `cudnn.allow_tf32=False`, `matmul.allow_tf32=False`, `model.eval()`, `torch.inference_mode()`, FP32, random tensors.
Latency = median of 20 CUDA-event timed forwards (5 warmup). Memory = `max_memory_allocated()` after `reset_peak_memory_stats()` (the live input counts, since the caller holds it). Energy = NVML **energy counter** (mJ) around a ≈1.5 s window of back-to-back forwards, divided by the iteration count (fallback: integrate `power_usage` at 100 Hz). OOM rows are caught and recorded as `OOM`.

## Fitted parameters (GTX 1650)

All θ are **effective parameters** fitted to the measurements: they absorb the model's simplifications and are correlated with each other — they are not separately identifiable hardware constants.

| θ | value | meaning |
|---|-------|---------|
| P | **6.11·10¹² flop/s** | effective FP32-throughput parameter — 2.05× the hardware peak (896 cores × 2 × 1.665 GHz ≈ 2.98 TFLOP/s); it compensates the additive per-op roofline. Clamping P to the peak refits BW to 214 GB/s at the same accuracy → (P, BW) are not identifiable separately |
| BW | **83.6 GB/s** | effective bandwidth parameter (spec 128 GB/s; the fit lands below spec because the byte model overcounts DRAM traffic — L2 hits are not modelled) |
| t_launch | **79.2 µs** | per-kernel launch overhead under WDDM; with the N = 23 assumption this gives a floor of 23·t_launch ≈ **1.82 ms** (only the product is identified; the real kernel count was not verifiable — no CUPTI on Windows) |
| c0 | **0.464 J** | regression intercept, *not* a physical constant: the effective fixed term c0 + c_b·(weights) ≈ **0.083 J** is what matches the measured small-workload energy |
| c_f | **2.77 nJ/flop** | effective — ~20–80× the physical J/flop implied by the measurements (≈0.03–0.14 nJ/flop); inflated by the collinearity with the Bytes term |
| c_b | **−91.2 nJ/byte** (−9.12·10⁻⁸ J/byte) | negative — a collinearity artifact (r = 1.000 between FLOPs and Bytes on this grid); the combination still predicts validation energy with 9.7 % MAPE |

Non-negative alternative (reported by `calibrate.py`): `E = c0 + P_idle·T + c_f·F + c_b·D` fitted with coefficients constrained ≥ 0 → P_idle ≈ **28.2 W**, c_f ≈ **0.025 nJ/flop**, c_b = 0, c0 ≈ 0.026 J; MAPE 10.6 % on all rows — comparable accuracy with sane signs and magnitudes. Non-negativity is a useful constraint, not a proof of physical meaning: these are still fitted (effective) values.

Quality (MAPE, %): **latency train 14.8 / validation 16.5**; **energy train 11.7 / validation 9.7**; **memory (analytical lower bound) train 44.2 / validation 38.1**. Excluding the **8 suspected-WDDM-spillover configs** (measured peak allocation > 2.5 GB; see the WDDM section): latency **11.7 / 13.0**, energy **9.1 / 6.9**. Measured/ideal memory ratio: median **1.60×** (p10 1.21, p90 3.21).

## Figures

Every plot shows the measured points together with the model prediction (train and validation points distinguished), with labelled axes and units.

| Figure | What it shows |
|---|---|
| [`latency_vs_S.png`](results/figures/latency_vs_S.png) | measured vs predicted latency: curves vs S for B ∈ {1, 8, 32, 128} + predicted (S, B) surface, seconds |
| [`latency_parity.png`](results/figures/latency_parity.png) | predicted vs measured latency parity, log-log (ms), train/validation MAPE in the title |
| [`regimes.png`](results/figures/regimes.png) | left: launch / memory / compute components of the model at B=32; right: analytical FLOPs ÷ measured latency (achieved TFLOP/s) vs workload with the calibrated P — the FLOPs-side check |
| [`memory.png`](results/figures/memory.png) | peak memory: ideal-allocator prediction vs measured `max_memory_allocated` (MB) + parity plot |
| [`energy.png`](results/figures/energy.png) | energy per forward vs S (J) + log-log parity |
| [`oom_boundary.png`](results/figures/oom_boundary.png) | memory-feasibility map: where the equation exceeds VRAM, with measured ok/OOM points |

## Results summary and 1-page discussion

132 configurations (11 sizes × 12 batches), **0 OOM rows** — on this Windows/WDDM box the driver silently pages oversized allocations into shared system memory instead of failing (why, and how to get real OOM rows: see *Windows/WDDM and OOM* below). Four findings:

1. **Performance regimes** (`regimes.png`, `latency_vs_S.png`): at small B·S² the 23 fixed launches dominate (measured floor ≈ 1.8–2.7 ms ≈ 23·t_launch); at mid sizes the summed `bytes/BW` terms win; the largest workloads push the 4 big convolutions to the compute side. The other 19 ops stay memory-bound at every (S, B), and since both FLOPs and Bytes scale as B·S², the aggregate log-log slope stays ≈ 2 — the crossover changes which term dominates, not the growth law. Achieved throughput saturates at ≈ 2.1 TFLOP/s (vs ≈ 3.0 TFLOP/s hardware peak) and drops on the suspected-spillover configs.
2. **Memory equation is a parallel lower bound** (`memory.png`): measured/ideal = 1.21–3.21× (median 1.60×) — cuDNN workspaces are allocated through the caching allocator (so they *are* counted in `max_memory_allocated`) and blocks are rounded up; the gap grows with S·B (wider-workspace algorithms for bigger tiles). The 19·B·S² law itself matches the ideal tensor lifecycle (live input included).
3. **Energy is affine in work and validates the best of the four** (`energy.png`, 9.7 % MAPE, parity within ~10 % over 3 decades). FLOPs and Bytes are collinear on this grid (r = 1.000), so c_f and c_b are regression artifacts (c_b < 0): no physical compute/DRAM split can be read off them. At small workloads the effective fixed term ≈ 0.083 J makes the energy per forward nearly independent of (S, B).
4. **Where it breaks:** the 8 suspected-spillover configs carry the largest errors — per-config latency 39–84 %, energy 25–73 % (six of the eight exceed 69 % latency error) — and dominate the MAPE budget; excluding them, latency drops to 11.7/13.0 % and energy to 9.1/6.9 %. The remaining error budget: noisy WDDM launch overhead at B=1; kernel-choice step-changes under `cudnn.benchmark=False`; NVML board energy including the Windows desktop (~10 % scatter); the memory bound predicting scale rather than the exact OOM point.

## Windows/WDDM and OOM

By default the NVIDIA driver on Windows silently spills CUDA allocations that do not fit in VRAM into shared system memory: no `OutOfMemoryError` is raised, the affected configs just slow down. **Actual shared-GPU-memory usage was not measured**: the 8 configs with measured peak allocation > 2.5 GB ((352,256), (384,256), (400,167), (400,256), (512,98), (512,128), (512,167), (512,256)) are flagged as *suspected* WDDM-spillover — the observable symptom being that they run 1.6–6.2× slower than the roofline ((512,256): measured 3.25 s vs 0.52 s predicted; there the equation gives 5.1 GB > 4 GiB VRAM and the measured peak is 6.2 GB).

**Why:** under WDDM the OS video-memory manager (VidMm) treats VRAM as a *paged* surface — a CUDA allocation is a WDDM resource that the OS can migrate between VRAM and system memory, and the driver's CUDA sysmem fallback (default policy on GeForce, driver ≥ 536.40) places allocations in shared system memory instead of failing (the Task-Manager counter is "Shared GPU memory"). This GPU also drives the display, and GeForce cards cannot switch to TCC — the Windows driver mode with dedicated, non-paged memory and honest OOM. The price of a spilled page is a PCIe transfer (~12–15 GB/s here vs 128 GB/s GDDR5) plus residency-management overhead — exactly the observed 1.6–6.2× slowdowns. Definitive proof would be sampling shared-GPU-memory usage during a (512,256) run (Task Manager / the `GPU Process Memory` performance counter); we did not measure it, hence *suspected*.

About the "4.3 GB" VRAM wall in `oom_boundary.png`: that line is `torch.cuda.get_device_properties(0).total_memory` = 4 294 508 544 bytes = 4.00 GiB (marketed "4 GB"; the same number is ≈ 4.29 in decimal GB). Real paging starts *before* the equation crosses this line, because the Windows desktop shares the same 4 GiB.

Two ways to obtain real OOM rows for the assignment's OOM-comparison requirement:

1. `uv run python measure.py --vram-cap 0.9` — an **artificial software-limit experiment**: the PyTorch caching allocator is capped at 90 % of VRAM, and allocations beyond the cap raise `torch.cuda.OutOfMemoryError`. An OOM row then means "above the cap", not necessarily physical-VRAM exhaustion (and the recorded `memory_bytes` never exceeds the cap).
2. NVIDIA Control Panel → Manage 3D Settings → **CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback** (driver ≥ 536.40; this box runs 572.16) — the driver refuses the allocation instead of spilling, so OOM reflects the real available VRAM.

The cleanest OOM data still comes from Linux/Colab (T4), where the paging path does not exist.

## Note on hw1_handwritten.pdf

`hw1_handwritten.pdf` is the scanned hand-written Part 1 derivation (3 pages): the four closed-form functions with the layer tables, the peak-memory argument, and the explicitly stated assumptions (A1–A6 for latency, E1–E3 for energy). The numbers are those of the analytical model in this README, reproduced exactly by `equations.py` (FLOPs(224,1) = 892 292 352; weights + BN buffers = 1 045 316 floats; traffic 133·S² + 2 148 per sample).
