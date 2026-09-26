"""calibrate.py — fit the GPU-dependent parameters theta (latency + energy).

Latency model (equations.latency):
    T = sum_ops max(flops_op / P, bytes_op / BW) + N * t_launch
    fitted params: P, BW, t_launch — EFFECTIVE parameters (P and BW are
    strongly correlated, only the product N*t_launch is identified).

Energy model (equations.energy):
    E = c0 + c_f * FLOPs + c_b * BytesMoved
    fitted params: c0, c_f, c_b — plain linear least squares; effective
    regression coefficients (FLOPs/Bytes are collinear, c_b may be < 0).
    A physically interpretable variant E = P_idle*T + c_f*F + c_b*D with
    non-negative coefficients (scipy.optimize.nnls) is reported alongside.

Suspected WDDM spillover: on Windows the CUDA driver silently pages
allocations that do not fit in VRAM into shared system memory, so no
OutOfMemoryError is raised; instead those configs run several times slower
than the roofline. Actual shared-memory usage is NOT measured; configs are
flagged by a proxy criterion (measured peak allocation > 2.5 GB on this
4 GiB GPU) and the quality metrics are additionally reported with them
excluded.

Training data : rows with is_validation == 0 and status ok
Validation    : rows with is_validation == 1 (never seen by the fit)

Writes results/theta.json.
"""

import json
import platform
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit, nnls

import equations as eq


def mape(pred, real):
    return float(np.mean(np.abs(pred - real) / real) * 100.0)


def fit_latency(train_S, train_B, train_lat):
    X = np.vstack([train_S, train_B])                   # (2, N)

    def flat(x, P, BW, t_launch):
        theta = {"P": P, "BW": BW, "t_launch": t_launch}
        return eq.latency(x[0], x[1], theta)

    p0 = [5e11, 1e11, 3e-5]
    bounds = ([1e9, 1e9, 1e-8], [1e14, 1e12, 5e-3])
    popt, _ = curve_fit(flat, X, train_lat,
                        p0=p0, bounds=bounds, sigma=train_lat,
                        absolute_sigma=False, maxfev=20000)
    return popt


def fit_energy(train_S, train_B, train_E):
    F = eq.flops(train_S, train_B)
    Mb = eq.bytes_moved(train_S, train_B)
    A = np.stack([np.ones_like(F), F, Mb], axis=1)
    # relative-error weighting: divide each row by its measured energy
    W = A / train_E[:, None]
    y = np.ones_like(train_E)
    coef, *_ = np.linalg.lstsq(W, y, rcond=None)
    return coef


def main():
    df = pd.read_csv("results/measurements.csv")
    num = pd.to_numeric(df["latency_s"], errors="coerce")
    ok = df.assign(latency_s=num).dropna(subset=["latency_s"])
    train = ok[ok["is_validation"] == 0]
    valid = ok[ok["is_validation"] == 1]

    # NOTE: the fits below use all train rows (WDDM-spillover rows included);
    # the relative weighting already makes their influence small (their
    # measured values are the largest on the grid).
    tS, tB = train["S"].to_numpy(float), train["B"].to_numpy(float)
    popt = fit_latency(tS, tB, train["latency_s"].to_numpy(float))
    theta_lat = {"P": float(popt[0]), "BW": float(popt[1]),
                 "t_launch": float(popt[2])}

    ecoef = fit_energy(tS, tB,
                       pd.to_numeric(train["energy_J"]).to_numpy(float))
    theta_en = {"c0": float(ecoef[0]), "c_f": float(ecoef[1]),
                "c_b": float(ecoef[2])}

    # quality on train / validation
    pred_tr = eq.latency(tS, tB, theta_lat)
    pred_va = eq.latency(valid["S"].to_numpy(float),
                         valid["B"].to_numpy(float), theta_lat)
    lat_q = {"train_mape_pct": mape(pred_tr, train["latency_s"].to_numpy(float)),
             "validation_mape_pct": mape(pred_va,
                                         valid["latency_s"].to_numpy(float))}

    e_tr = pd.to_numeric(train["energy_J"]).to_numpy(float)
    e_va = pd.to_numeric(valid["energy_J"]).to_numpy(float)
    en_q = {"train_mape_pct": mape(eq.energy(tS, tB, theta_en), e_tr),
            "validation_mape_pct": mape(eq.energy(valid["S"].to_numpy(float),
                                                  valid["B"].to_numpy(float),
                                                  theta_en), e_va)}

    m_tr = pd.to_numeric(train["memory_bytes"]).to_numpy(float)
    m_pred_tr = eq.memory(tS, tB)
    mem_q = {"train_mape_pct": mape(m_pred_tr, m_tr)}
    m_va = pd.to_numeric(valid["memory_bytes"]).to_numpy(float)
    mem_q["validation_mape_pct"] = mape(
        eq.memory(valid["S"].to_numpy(float), valid["B"].to_numpy(float)), m_va)

    # --- effective energy alternative: E = c0 + P_idle*T + c_f*F + c_b*D,
    #     all coefficients >= 0 (physically interpretable; nnls)
    F_ok = eq.flops(ok["S"].to_numpy(float), ok["B"].to_numpy(float))
    D_ok = eq.bytes_moved(ok["S"].to_numpy(float), ok["B"].to_numpy(float))
    T_ok = ok["latency_s"].to_numpy(float)
    E_ok = pd.to_numeric(ok["energy_J"]).to_numpy(float)
    A_alt = np.stack([np.ones_like(F_ok), T_ok, F_ok, D_ok], axis=1)
    coef_nn, _ = nnls(A_alt / E_ok[:, None], np.ones_like(E_ok))
    en_alt = {
        "model": "E = c0 + P_idle*T + c_f*F + c_b*D (nnls, all coefficients >= 0)",
        "c0_J": float(coef_nn[0]),
        "P_idle_W": float(coef_nn[1]),
        "c_f_J_per_flop": float(coef_nn[2]),
        "c_b_J_per_byte": float(coef_nn[3]),
        "mape_all_rows_pct": mape(A_alt @ coef_nn, E_ok),
    }

    # --- Suspected WDDM spillover: on Windows the driver silently pages
    #     allocations beyond VRAM into shared system memory, so no OOM is
    #     raised; these configs instead run 1.6-6.2x slower than the roofline.
    #     Shared-memory usage itself is NOT measured -> the flag is a proxy.
    SPILL_BYTES = 2.5e9                       # empirical threshold, 4 GiB GPU
    mem_num = pd.to_numeric(ok["memory_bytes"], errors="coerce")
    spill_mask = mem_num > SPILL_BYTES
    sub = ok[~spill_mask]
    sub_tr = sub[sub["is_validation"] == 0]
    sub_va = sub[sub["is_validation"] == 1]
    spill_q = {
        "criterion": f"suspected WDDM spillover: measured peak allocation > {SPILL_BYTES:.1e} bytes (proxy; shared-memory usage itself was not measured)",
        "configs": [[int(r.S), int(r.B)] for r in ok[spill_mask].itertuples()],
        "latency_mape_pct_excl_spillover": {
            "train": mape(eq.latency(sub_tr["S"].to_numpy(float),
                                    sub_tr["B"].to_numpy(float), theta_lat),
                         sub_tr["latency_s"].to_numpy(float)),
            "validation": mape(eq.latency(sub_va["S"].to_numpy(float),
                                          sub_va["B"].to_numpy(float), theta_lat),
                               sub_va["latency_s"].to_numpy(float)),
        },
        "energy_mape_pct_excl_spillover": {
            "train": mape(eq.energy(sub_tr["S"].to_numpy(float),
                                    sub_tr["B"].to_numpy(float), theta_en),
                          pd.to_numeric(sub_tr["energy_J"]).to_numpy(float)),
            "validation": mape(eq.energy(sub_va["S"].to_numpy(float),
                                        sub_va["B"].to_numpy(float), theta_en),
                               pd.to_numeric(sub_va["energy_J"]).to_numpy(float)),
        },
    }

    # OOM agreement: predicted memory vs the biggest allocation that failed
    oom = df[df["latency_s"].astype(str) == "OOM"].copy()
    free_b = torch.cuda.get_device_properties(0).total_memory
    pred_oom = []
    for r in oom.itertuples():
        pred_oom.append(float(eq.memory(r.S, r.B)) > free_b * 0.80)
    oom_q = {"n_oom_rows": int(len(oom)),
             "n_predicted_oom_by_equation": int(sum(pred_oom))}

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "total_gpu_memory_bytes": int(free_b),
        "n_kernels_per_forward_equation": eq.N_KERNELS,
        "latency": theta_lat,
        "energy": theta_en,
        "quality": {"latency": lat_q, "energy": en_q, "memory": mem_q,
                    "oom": oom_q, "wddm_spillover": spill_q},
        "energy_effective_alt": en_alt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("results/theta.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
