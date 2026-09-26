"""measure.py — Part 3: measurement protocol for Homework 1.

Runs the SmallCNN forward pass on the real GPU over the (S, B) grid and
records, for each configuration:

    S, B, is_validation, latency_s (median), memory_bytes (peak
    max_memory_allocated during a forward), energy_J (net GPU energy per
    forward, from NVML), and status (ok / OOM).

Also emits results/kernels.csv with the CUDA kernel name per labelled layer
for a representative subset, via torch.profiler + a chrome-trace parse.

Windows/WDDM note: by default the NVIDIA driver silently spills CUDA
allocations that do not fit in VRAM into shared system memory, so
OutOfMemoryError may never fire — the affected configs just run several
times slower.  Use --vram-cap 0.9 to force real OOM rows (PyTorch allocator
cap), or disable the driver's sysmem fallback (NVIDIA Control Panel ->
Manage 3D Settings -> CUDA - Sysmem Fallback Policy -> Prefer No Sysmem
Fallback).

Reproduce:
    uv run python measure.py            # full run -> results/measurements.csv
    uv run python measure.py --quick     # small smoke subset

Measurement flags are fixed exactly as the assignment specifies.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import tempfile
import time

import numpy as np
import torch
import pynvml

import models

# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 2026

# configs for which results/kernels.csv rows are dumped (spans all 3 regimes)
PROFILE_SET = {(32, 1), (32, 32), (64, 8), (128, 1), (128, 32),
               (224, 16), (256, 1), (256, 32), (512, 1), (512, 8)}


def build_grid():
    rng = np.random.default_rng(SEED)
    # 4 extra multiples of 16 in [32, 512] not already in BASE_S
    mult16 = [s for s in range(32, 513, 16)]
    extra_s_pool = [s for s in mult16 if s not in BASE_S]
    extra_S = sorted(rng.choice(extra_s_pool, size=4, replace=False).tolist())
    # 3 extra batch sizes in [1, 256] not powers of two
    pow2 = {2 ** i for i in range(9)}                     # 1..256
    extra_b_pool = [b for b in range(1, 257) if b not in pow2]
    extra_B = sorted(rng.choice(extra_b_pool, size=3, replace=False).tolist())
    S_vals = sorted(BASE_S + extra_S)
    B_vals = sorted(BASE_B + extra_B)
    val_S = set(extra_S)
    val_B = set(extra_B)
    return S_vals, B_vals, val_S, val_B


# ---------------------------------------------------------------------------
# energy measurement via NVML
# ---------------------------------------------------------------------------
class EnergyMeter:
    """Net GPU energy per forward, in joules.

    Uses the cumulative energy counter (millijoules) when the driver exposes
    it; otherwise integrates reported board power over a polling window.
    """

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.use_counter = True
        try:
            self._read_counter()
        except Exception:
            self.use_counter = False

    def _read_counter(self):
        field = pynvml.NVML_FI_DEV_ENERGY_POWER
        vals = pynvml.nvmlDeviceGetFieldValues(self.handle, [field])
        return vals[0].value.llVal                          # mJ

    def _read_power(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle)  # mW

    def measure(self, fn, target_s=1.5):
        """Call fn() repeatedly for ~target_s seconds; return (J/iter, n_iters)."""
        if self.use_counter:
            e0 = self._read_counter()
        else:
            e0 = 0.0
            power_samples = []

        fn()                                                 # ensure kernels warm
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        n = 0
        if self.use_counter:
            e0 = self._read_counter()
            while time.perf_counter() - t_start < target_s:
                fn()
                n += 1
            torch.cuda.synchronize()
            dur = time.perf_counter() - t_start
            e1 = self._read_counter()
            dE = (e1 - e0) / 1000.0                          # mJ -> J
            return dE / n, n, dur
        else:
            import threading
            stop = False

            def poll():
                while not stop:
                    power_samples.append((time.perf_counter(), self._read_power()))
                    time.sleep(0.01)

            th = threading.Thread(target=poll)
            th.start()
            while time.perf_counter() - t_start < target_s:
                fn()
                n += 1
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            stop = True
            th.join()
            # integrate power over time -> J, subtract estimated idle baseline
            # (baseline handled by calibrate model; here return raw avg power*time)
            total = 0.0
            ts = power_samples
            for i in range(len(ts) - 1):
                dt = ts[i + 1][0] - ts[i][0]
                total += (ts[i][1] / 1000.0) * dt           # W * s = J
            return total / n, n, (t_end - t_start)


# ---------------------------------------------------------------------------
# per-config measurement
# ---------------------------------------------------------------------------
def set_flags():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def bench_latency(model, x, warmup=5, iters=20):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) / 1000.0)    # ms -> s
    return statistics.median(timings)


def bench_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return peak                                             # absolute peak


def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--profile-subset", type=int, default=10,
                    help="number of configs to dump kernel names for")
    ap.add_argument("--vram-cap", type=float, default=None,
                    help="cap the PyTorch caching allocator at this fraction of "
                         "total VRAM so allocations beyond it raise a real "
                         "torch.cuda.OutOfMemoryError (WDDM would otherwise "
                         "silently spill them into shared system memory); e.g. 0.9")
    args = ap.parse_args()

    set_flags()
    if args.vram_cap is not None:
        torch.cuda.set_per_process_memory_fraction(args.vram_cap, 0)
        tot = torch.cuda.get_device_properties(0).total_memory
        print(f"allocator capped at {args.vram_cap:.0%} of VRAM "
              f"({tot * args.vram_cap / 2**30:.2f} GiB) -> real OOM rows expected")
    device = "cuda"
    model = models.build_model(device)
    S_vals, B_vals, val_S, val_B = build_grid()
    if args.quick:
        S_vals, B_vals = [32, 128, 256], [1, 8, 32]
        val_S, val_B = set(), set()

    meter = EnergyMeter()
    os.makedirs("results", exist_ok=True)
    rows = []
    n_prof = 0
    kernel_rows = []

    total = len(S_vals) * len(B_vals)
    i = 0
    for S in S_vals:
        for B in B_vals:
            i += 1
            tag = f"[{i}/{total}] S={S} B={B}"
            try:
                with torch.inference_mode():
                    x = torch.randn(B, 3, S, S, device=device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if is_oom(e):
                    torch.cuda.empty_cache()
                    rows.append((S, B, int(S in val_S or B in val_B),
                                 "OOM", "OOM", "OOM"))
                    print(f"{tag} OOM (input)")
                    continue
                raise

            fwd = lambda: model(x)
            with torch.inference_mode():
                # memory
                try:
                    mem = bench_memory(model, x)
                    # latency
                    lat = bench_latency(model, x) if not args.quick else \
                        bench_latency(model, x, warmup=3, iters=10)
                    # energy
                    en, en_iters, en_dur = meter.measure(fwd,
                                                         target_s=0.8 if args.quick else 1.5)
                    status = "ok"
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if is_oom(e):
                        torch.cuda.empty_cache()
                        rows.append((S, B, int(S in val_S or B in val_B),
                                     "OOM", "OOM", "OOM"))
                        print(f"{tag} OOM (forward)")
                        continue
                    raise

            is_val = int(S in val_S or B in val_B)
            rows.append((S, B, is_val, lat, mem, en))
            print(f"{tag} lat={lat*1e3:.3f}ms mem={mem/1e6:.1f}MB "
                  f"energy={en:.3f}J  (val={is_val})")

            # kernel profiling for the representative subset
            if not args.quick and (S, B) in PROFILE_SET:
                try:
                    kr = profile_kernels(model, x, S, B)
                    kernel_rows.extend(kr)
                    n_prof += 1
                except Exception as pe:
                    print(f"   profile skipped: {pe}")
            del x
            torch.cuda.empty_cache()

    # write measurements.csv (column order per the assignment:
    # S, B, latency, memory or OOM, energy, is_validation)
    with open("results/measurements.csv", "w", newline="\n") as f:
        f.write("S,B,latency_s,memory_bytes,energy_J,is_validation\n")
        for (S, B, v, lat, mem, en) in rows:
            f.write(f"{S},{B},{lat},{mem},{en},{v}\n")
    print("wrote results/measurements.csv",
          f"({sum(1 for r in rows if r[3]=='OK' or isinstance(r[3], float))} ok, "
          f"{sum(1 for r in rows if r[3]=='OOM')} OOM)")

    if kernel_rows:
        with open("results/kernels.csv", "w", newline="\n") as f:
            f.write("S,B,layer,kernel\n")
            for (S, B, layer, kernel) in kernel_rows:
                f.write(f"{S},{B},{layer},\"{kernel}\"\n")
        print("wrote results/kernels.csv")


# ---------------------------------------------------------------------------
# kernel-name extraction (torch.profiler)
#
# NOTE: on Windows CUPTI is unavailable, so the chrome trace has no
# `cat == "kernel"` events with the driver-level kernel names (e.g.
# sm75_xxx_...).  Fallback: use the FunctionEvents with nonzero
# self_device_time and report the CUDA op name that launched the kernel
# (aten::cudnn_convolution, aten::cudnn_batch_norm, ...) together with its
# duration.  On Linux/Colab the same code path returns true kernel names
# from the trace events.
# ---------------------------------------------------------------------------
def profile_kernels(model, x, S, B):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()

    # -- primary path: chrome trace kernel events (Linux / CUPTI available) --
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        path = tf.name
    try:
        prof.export_chrome_trace(path)
        with open(path) as fh:
            trace = json.load(fh)
        events = trace.get("traceEvents", [])
        anns = [e for e in events if e.get("cat") == "user_annotation"
                and e.get("ph") == "X"]
        kern = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        if kern:
            out = []
            for k in kern:
                kts, kdur = k["ts"], k.get("dur", 0)
                best = None
                for a in anns:
                    ats, adur = a["ts"], a.get("dur", 0)
                    if ats <= kts and kts <= ats + adur:
                        if best is None or (adur < best[1] - best[0]):
                            best = (a["name"], adur)
                layer = best[0] if best else "?"
                out.append((S, B, layer, k["name"][:60]))
            return out
    finally:
        if os.path.exists(path):
            os.remove(path)

    # -- fallback path: FunctionEvents with self device time (Windows) -------
    out = []
    seen = set()
    for e in prof.events():
        dt = getattr(e, "self_device_time_total", 0) or 0
        if dt > 0 and e.name not in seen:
            seen.add(e.name)
            out.append((S, B, e.name, f"n/a (no CUPTI on Windows; self_device_time={dt:.0f}us)"))
    return out


if __name__ == "__main__":
    main()
