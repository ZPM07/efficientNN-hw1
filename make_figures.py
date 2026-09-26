"""make_figures.py — Part 5: predicted vs measured plots for Homework 1.

Every figure overlays measured points (train and validation markers) on the
equations' prediction curve/surface.  Writes results/figures/*.png and
prints error tables for the README.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import equations as eq

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 9})

MARK_OK = dict(marker="o", ms=4.5, ls="none")


def load():
    df = pd.read_csv("results/measurements.csv")
    for col in ["latency_s", "memory_bytes", "energy_J"]:
        df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = np.where(df["latency_s_num"].notna(), "ok", "OOM")
    with open("results/theta.json") as f:
        theta = json.load(f)
    return df, theta


def split_sets(df):
    ok = df[df.status == "ok"]
    tr = ok[ok.is_validation == 0]
    va = ok[ok.is_validation == 1]
    return tr, va


def scatter_trva(ax, S, y_tr, y_va, label=None, **kw):
    ax.scatter(S, y_tr, color="#1f77b4", alpha=0.75,
               label="measured (train)" if label else None, **MARK_OK)
    ax.scatter(S, y_va, color="#d62728", alpha=0.9, marker="X", ms=7,
               label="measured (validation)" if label else None)


def main():
    os.makedirs("results/figures", exist_ok=True)
    df, theta = load()
    thL, thE = theta["latency"], theta["energy"]
    tr, va = split_sets(df)

    # ------------------------------------------------------------------
    # 1. latency vs S for selected batch sizes + surface
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for B in (1, 8, 32, 128):
        x = np.linspace(32, 512, 120)
        axes[0].plot(x, eq.latency(x, B, thL), lw=1.4, label=f"pred B={B}")
        m_tr = tr[tr.B == B]
        m_va = va[va.B == B]
        if len(m_tr):
            axes[0].scatter(m_tr.S, m_tr.latency_s_num, s=18, alpha=0.7,
                            color=axes[0].lines[-1].get_color())
        if len(m_va):
            axes[0].scatter(m_va.S, m_va.latency_s_num, s=26, marker="X",
                            color=axes[0].lines[-1].get_color())
    axes[0].set(xlabel="image size S (px)", ylabel="forward latency (s)",
                yscale="log", title="Latency vs S (curves = model, dots = measured)")
    axes[0].legend(fontsize=7)

    ax3 = fig.add_subplot(133, projection="3d")
    SS, BB = np.meshgrid(np.linspace(32, 512, 24), [1, 2, 4, 8, 16, 32, 64])
    ZZ = eq.latency(SS, BB, thL)
    ax3.plot_surface(SS, BB, ZZ, alpha=0.45, cmap="viridis")
    ax3.scatter(tr.S, tr.B, tr.latency_s_num, color="k", s=8, label="train")
    ax3.scatter(va.S, va.B, va.latency_s_num, color="r", s=14, marker="X",
                label="validation")
    ax3.set(xlabel="S", ylabel="B", zlabel="latency (s)", title="Latency surface")
    ax3.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/latency_vs_S.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. memory vs S  + parity
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for B in (1, 8, 32, 128):
        x = np.linspace(32, 512, 120)
        pred = eq.memory(x, B) / 1e6
        axes[0].plot(x, pred, lw=1.4, label=f"predicted B={B}")
        m = tr[tr.B == B]
        if len(m):
            axes[0].scatter(m.S, m.memory_bytes_num / 1e6, s=18, alpha=0.7,
                            color=axes[0].lines[-1].get_color())
        m = va[va.B == B]
        if len(m):
            axes[0].scatter(m.S, m.memory_bytes_num / 1e6, s=26, marker="X",
                            color=axes[0].lines[-1].get_color())
    axes[0].axhline(theta["total_gpu_memory_bytes"] / 1e6, color="k", ls="--",
                    lw=1, label="physical VRAM (OOM wall)")
    axes[0].set(xlabel="image size S (px)", ylabel="peak allocated (MB)",
                title="Peak memory: ideal-allocator prediction vs measured")
    axes[0].legend(fontsize=7)

    allpred = eq.memory(tr.S, tr.B) / 1e6
    axes[1].scatter(allpred, tr.memory_bytes_num / 1e6, s=16, alpha=0.6,
                    label="train")
    av = eq.memory(va.S, va.B) / 1e6
    axes[1].scatter(av, va.memory_bytes_num / 1e6, s=26, marker="X",
                    color="r", alpha=0.9, label="validation")
    lim = max(allpred.max(), tr.memory_bytes_num.max() / 1e6) * 1.1
    axes[1].plot([0, lim], [0, lim], "k--", lw=1)
    axes[1].set(xlabel="predicted memory (MB)", ylabel="measured memory (MB)",
                title="Memory parity (model is a systematic lower bound)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/figures/memory.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. energy parity + vs S
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for B in (1, 32):
        x = np.linspace(32, 512, 120)
        axes[0].plot(x, eq.energy(x, B, thE), lw=1.4, label=f"predicted B={B}")
        m = tr[tr.B == B]
        if len(m):
            axes[0].scatter(m.S, m.energy_J_num, s=18, alpha=0.7,
                            color=axes[0].lines[-1].get_color())
        m = va[va.B == B]
        if len(m):
            axes[0].scatter(m.S, m.energy_J_num, s=26, marker="X",
                            color=axes[0].lines[-1].get_color())
    axes[0].set(xlabel="image size S (px)", ylabel="energy per forward (J)",
                title="GPU energy: model vs measured")
    axes[0].legend()

    axes[1].scatter(eq.energy(tr.S, tr.B, thE), tr.energy_J_num, s=16,
                    alpha=0.6, label="train")
    axes[1].scatter(eq.energy(va.S, va.B, thE), va.energy_J_num, s=26,
                    marker="X", color="r", alpha=0.9, label="validation")
    lim = max(eq.energy(tr.S, tr.B, thE).max(), va.energy_J_num.max()) * 1.15
    axes[1].plot([0, lim], [0, lim], "k--", lw=1)
    axes[1].set(xlabel="predicted energy (J)", ylabel="measured energy (J)",
                title="Energy parity", xscale="log", yscale="log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/figures/energy.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. latency parity
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    p_tr = eq.latency(tr.S, tr.B, thL) * 1e3
    p_va = eq.latency(va.S, va.B, thL) * 1e3
    ax.scatter(p_tr, tr.latency_s_num * 1e3, s=16, alpha=0.6, label="train")
    ax.scatter(p_va, va.latency_s_num * 1e3, s=26, marker="X", color="r",
               alpha=0.9, label="validation")
    lim = max(p_tr.max(), p_va.max(), (tr.latency_s_num * 1e3).max()) * 1.2
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    lo = min(p_tr.min(), (tr.latency_s_num * 1e3).min()) * 0.7
    ax.set(xscale="log", yscale="log", xlim=[lo, lim], ylim=[lo, lim],
           xlabel="predicted latency (ms, log)",
           ylabel="measured latency (ms, log)",
           title=f"Latency parity — MAPE train {theta['quality']['latency']['train_mape_pct']:.0f}%"
                 f" / val {theta['quality']['latency']['validation_mape_pct']:.0f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/latency_parity.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. regime breakdown + achieved TFLOPS
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    B = 32
    x = np.linspace(32, 512, 120)
    launch = eq.N_KERNELS * thL["t_launch"] * np.ones_like(x)
    # per-op summed components with each op taking only one roofline side
    def comp(kind):
        tot = np.zeros_like(x)
        for k, (name, iu, ic, ou, oc, w, nk, ck) in enumerate(eq._OPS):
            f_op = B * (eq._FU[k] * x ** 2 + eq._FC[k])
            b_op = 4.0 * (B * ((iu + ou) * x ** 2 + ic + oc) + w)
            tot += (f_op / thL["P"]) if kind == "compute" else (b_op / thL["BW"])
        return tot
    axes[0].plot(x, comp("memory"), label="sum bytes/BW (memory side)")
    axes[0].plot(x, comp("compute"), label="sum flops/P (compute side)")
    axes[0].plot(x, launch * np.ones_like(x), label=f"{eq.N_KERNELS} launches")
    axes[0].plot(x, eq.latency(x, B, thL), "k", lw=1.6, label="model max-of-rooflines")
    m = tr[tr.B == B]
    axes[0].scatter(m.S, m.latency_s_num, color="0.2", s=18, zorder=5,
                    label="measured B=32")
    axes[0].set(xscale="log", yscale="log", xlabel="image size S (px)",
                ylabel="latency (s)", title="Regime breakdown at B=32")
    axes[0].legend(fontsize=7)

    ok = df[df.status == "ok"]
    tf = eq.flops(ok.S, ok.B) / ok.latency_s_num / 1e12
    axes[1].scatter(ok.S * np.sqrt(ok.B), tf, c=ok.B, s=14, cmap="plasma")
    axes[1].axhline(thL["P"] / 1e12, color="k", ls="--", lw=1,
                    label=f"calibrated P = {thL['P']/1e12:.2f} TFLOP/s")
    axes[1].set(xlabel="S·√B (workload scale)",
                ylabel="achieved TFLOP/s",
                title="Achieved throughput vs workload (color = B)")
    axes[1].legend()
    fig.colorbar(axes[1].collections[0], ax=axes[1], label="batch B")
    fig.tight_layout()
    fig.savefig("results/figures/regimes.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. OOM boundary
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    SS, BB = np.meshgrid(np.linspace(32, 512, 250), np.linspace(1, 256, 250))
    P = eq.memory(SS, BB, ) / 1e6
    vram = theta["total_gpu_memory_bytes"] / 1e6
    ax.contourf(SS, BB, P, levels=[0, vram, 1e9], colors=["#bfe3c0", "#f3b8b8"])
    ax.scatter(df[df.status == "ok"].S, df[df.status == "ok"].B, s=10,
               color="#1f77b4", label="measured ok")
    om = df[df.status == "OOM"]
    if len(om):
        ax.scatter(om.S, om.B, s=18, marker="x", color="k", label="measured OOM")
    ax.set(xlabel="image size S (px)", ylabel="batch B",
           title=f"Memory feasibility: green = equation < {vram/1000:.1f} GB VRAM")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/oom_boundary.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # error tables for README
    # ------------------------------------------------------------------
    print(json.dumps(theta["quality"], indent=2))
    print("figures written to results/figures/")


if __name__ == "__main__":
    main()
