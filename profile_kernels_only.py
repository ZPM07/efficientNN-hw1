"""profile_kernels_only.py — regenerate results/kernels.csv without re-running
the full measurement grid (the measurement run predates the kernels fallback).

Run from hw1/:  uv run python profile_kernels_only.py
"""

import os

import torch

import models
from measure import PROFILE_SET, set_flags, profile_kernels

set_flags()
model = models.build_model("cuda")
os.makedirs("results", exist_ok=True)

rows = []
for (S, B) in sorted(PROFILE_SET):
    x = torch.randn(B, 3, S, S, device="cuda")
    try:
        kr = profile_kernels(model, x, S, B)
        rows.extend(kr)
        print(f"S={S} B={B}: {len(kr)} kernel rows")
    except Exception as e:
        print(f"S={S} B={B}: FAILED {e}")
    del x
    torch.cuda.empty_cache()

with open("results/kernels.csv", "w", newline="\n") as f:
    f.write("S,B,layer,kernel\n")
    for (S, B, layer, kernel) in rows:
        k = str(kernel).replace('"', "'")
        f.write(f'{S},{B},{layer},"{k}"\n')
print(f"wrote results/kernels.csv ({len(rows)} rows)")
