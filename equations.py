"""equations.py — closed-form analytical model of SmallCNN (Homework 1).

All four public functions accept Python scalars OR NumPy arrays for
`image_size` and `batch` (standard broadcasting) and return plain floats
or NumPy arrays.

Conventions (as derived on paper in hw1_handwritten.pdf):

FLOPs
    1 multiply-accumulate = 2 FLOPs.
    Conv:   2 * B * H_out * W_out * C_out * C_in * k^2
    BN (eval): 2 FLOPs per element (x*a + b, affine applied on the fly)
    ReLU:   1 FLOP per element (compare-select)
    MaxPool: counted as 0 FP-FLOPs (comparisons only: 8 compares per
             output element x 2S^2 output elements/sample = 16*B*S^2)
    GAP:    per channel, n = S^2/256 summed values cost n-1 adds + 1
            divide -> (2S^2 - 512) + 512 = 2S^2 FLOPs in total
    Linear: 2 * in * out per sample

Memory
    Peak of torch.cuda.max_memory_allocated() during one forward under
    inference_mode(): weights+BN buffers + everything live at the most
    crowded moment of the forward.  Under this convention the input tensor
    is alive during the whole forward (the caller holds a reference to it),
    so the peak stage is the first BatchNorm: network input 3S^2 + BN input
    8S^2 + BN output 8S^2 = 19S^2 floats per sample.  GAP output and logits
    appear later and never coexist with this peak.
    cuDNN workspaces are allocated through the PyTorch caching allocator,
    so they DO count towards max_memory_allocated: this equation is an
    ideal-allocator lower bound, not an exact prediction.

Bytes moved
    Every op reads its input and writes its output (ReLU in-place still
    reads + writes the same buffer); every conv/linear reads its weights once.

Latency  (theta = {P, BW, t_launch})
    T(S,B) = sum over ops of  max(flops_op / P,  bytes_op / BW)  +  N * t_launch
    P, BW, t_launch are EFFECTIVE parameters fitted to measurements, not
    separately identifiable hardware constants: P and BW are strongly
    correlated (clamping P to the hardware peak moves the fitted BW to
    ~214 GB/s at the same accuracy), N = 23 launches is an assumption (one
    kernel per high-level op), and only the product N*t_launch is identified.

Energy  (theta_energy = {c0, c_f, c_b})
    E(S,B) = c0 + c_f * FLOPs(S,B) + c_b * Bytes(S,B)      [joules]
    Plain linear-regression coefficients, NOT physical per-flop/per-byte
    costs: FLOPs and Bytes are collinear on this grid (r ~ 1.0), so c_f and
    c_b trade off (c_b can be negative) and c0 is a free intercept; only
    their combination is identified.  A physically interpretable variant
    E = P_idle*T + c_f*F + c_b*D with non-negative coefficients is reported
    by calibrate.py for comparison.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Architecture description.
#
# Sizes are expressed per sample as:
#   "u"  = units of S^2 (float elements):  elems = u * S^2
#   "c"  = constant number of float elements (independent of S)
# so the sums below are closed-form polynomials in S and B.
#
# Resolution chain: S -> S/2 -> S/4 -> S/4 -> S/8 -> S/8 -> S/16 -> S/16
# ----------------------------------------------------------------------------

# weights (floats): conv kernels + BN w/b + BN running stats + linear w/b
_CONV_W = (3 * 49 * 32) + (32 * 25 * 64) + (64 * 9 * 128) \
          + (128 * 1 * 256) + (256 * 9 * 256) + (256 * 1 * 512)
_LIN_W = 512 * 256 + 256 + 256 * 100 + 100
_BN_W = 4 * (32 + 64 + 128 + 256 + 256 + 512)   # weight, bias, run_mean, run_var
WEIGHT_ELEMS = _CONV_W + _LIN_W + _BN_W         # = 1_045_316 floats

# one op: (name, in_u, in_c, out_u, out_c, weight_elems_read, kernel_count, cin_k2)
# cin_k2 is nonzero only for convs: C_in * k^2, so conv MACs = out_elems * cin_k2.
_OPS = [
    #            in_u  in_c out_u out_c  w_read  n_k  cin_k2
    ("c1_conv",    3,    0,    8,    0,    4_704,  1,   3 * 49),
    ("c1_bn",      8,    0,    8,    0,      128,  1,     0),
    ("c1_relu",    8,    0,    8,    0,        0,  1,     0),
    ("pool",       8,    0,    2,    0,        0,  1,     0),
    ("c2_conv",    2,    0,    4,    0,   51_200,  1,  32 * 25),
    ("c2_bn",      4,    0,    4,    0,      256,  1,     0),
    ("c2_relu",    4,    0,    4,    0,        0,  1,     0),
    ("c3_conv",    4,    0,    2,    0,   73_728,  1,  64 * 9),
    ("c3_bn",      2,    0,    2,    0,      512,  1,     0),
    ("c3_relu",    2,    0,    2,    0,        0,  1,     0),
    ("c4_conv",    2,    0,    4,    0,   32_768,  1, 128 * 1),
    ("c4_bn",      4,    0,    4,    0,    1_024,  1,     0),
    ("c4_relu",    4,    0,    4,    0,        0,  1,     0),
    ("c5_conv",    4,    0,    1,    0,  589_824,  1, 256 * 9),
    ("c5_bn",      1,    0,    1,    0,    1_024,  1,     0),
    ("c5_relu",    1,    0,    1,    0,        0,  1,     0),
    ("c6_conv",    1,    0,    2,    0,  131_072,  1, 256 * 1),
    ("c6_bn",      2,    0,    2,    0,    2_048,  1,     0),
    ("c6_relu",    2,    0,    2,    0,        0,  1,     0),
    ("gap",        2,    0,    0,  512,        0,  1,     0),
    ("fc1",        0,  512,    0,  256,  131_328,  1,     0),
    ("fc1_relu",   0,  256,    0,  256,        0,  1,     0),
    ("fc2",        0,  256,    0,  100,   25_700,  1,     0),
]

# per-sample FLOP coefficients of each op:  flops = (fu*S^2 + fc)
_FU, _FC = [], []
for name, iu, ic, ou, oc, w, _nk, cin_k2 in _OPS:
    if name.endswith("_conv"):
        fu, fc = 2.0 * ou * cin_k2, 0.0            # 2 * out_elems * C_in*k^2
    elif name.endswith("_bn"):
        fu, fc = 2.0 * ou, 2.0 * oc                # eval BN: affine ~2 FLOPs/elem
    elif name.endswith("_relu"):
        fu, fc = 1.0 * ou, 1.0 * oc                # 1 FLOP/elem
    elif name == "pool":
        fu, fc = 0.0, 0.0                          # comparisons, not FLOPs
    elif name == "gap":
        fu, fc = 1.0 * iu, 0.0                   # (n-1 adds + 1 divide)/channel = iu*S^2 = 2S^2
    else:                                          # fc1, fc2 (Linear): 2*in*out/sample
        fu, fc = 0.0, 2.0 * ic * oc
    _FU.append(fu)
    _FC.append(fc)

# sanity: c1_conv: out=8*S^2 elems, C_in*k^2 = 3*49=147 -> fu = 2*8*147 = 2352
assert abs(_FU[0] - 2352.0) < 1e-9

_FLOPS_U = float(sum(_FU))          # 17777.0   FLOPs per sample per S^2
_FLOPS_C = float(sum(_FC))          # 313600.0  FLOPs per sample (S-independent)

# per-sample traffic (float element reads+writes):  elems = (mu*S^2 + mc)
_TRAFFIC_U = float(sum(o[1] + o[3] for o in _OPS))        # in_u + out_u
_TRAFFIC_C = float(sum(o[2] + o[4] for o in _OPS))        # in_c + out_c
_WEIGHT_ELEMS_READ = float(sum(o[5] for o in _OPS))       # weights read once per fwd
N_KERNELS = int(sum(o[6] for o in _OPS))                  # launch count per forward
assert _WEIGHT_ELEMS_READ == WEIGHT_ELEMS                 # every weight read once

assert _FLOPS_U == 17777.0 and _FLOPS_C == 313600.0, (_FLOPS_U, _FLOPS_C)
assert _TRAFFIC_U == 133.0, _TRAFFIC_U
assert N_KERNELS == 23, N_KERNELS


def _as(x):
    return np.asarray(x, dtype=float)


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

def flops(image_size, batch):
    """Floating-point operations of one forward pass (1 MAC = 2 FLOPs).

    FLOPs(S, B) = B * (17777 * S^2 + 313600)
    """
    S, B = _as(image_size), _as(batch)
    return B * (_FLOPS_U * S ** 2 + _FLOPS_C)


def bytes_moved(image_size, batch):
    """Estimated LOGICAL traffic of one forward pass, bytes (activations
    read+write + each weight read once).

    This is a modelling assumption, not literal DRAM traffic: weights and
    activations may stay in cache between ops/iterations, and a single conv
    may re-read its input several times.
    """
    S, B = _as(image_size), _as(batch)
    return 4.0 * (B * (_TRAFFIC_U * S ** 2 + _TRAFFIC_C) + _WEIGHT_ELEMS_READ)


def memory(image_size, batch):
    """Peak allocated GPU bytes during one forward pass (inference_mode, eval).

    Ideal refcounting allocator.  Under the measurement convention
    (max_memory_allocated over one forward) the input tensor is alive
    during the whole forward: the caller (measure.py) holds a reference to
    it, so it cannot be freed when c1_conv produces its output.  The most
    crowded moment is the first BatchNorm:

        network input 3S^2 (alive) + BN input 8S^2 + BN output 8S^2 = 19S^2

    floats per sample.  The GAP output (512B floats) and the logits (100B)
    appear much later, when only <= 5S^2 + O(B) floats per sample are live,
    so they never coexist with this peak and are not added.  cuDNN
    workspaces and caching-allocator round-ups are not modelled:
    workspaces are allocated through the PyTorch caching allocator and
    therefore DO appear in max_memory_allocated, making this equation a
    systematic lower bound rather than an exact prediction.
    """
    S, B = _as(image_size), _as(batch)
    peak_activation_units = 19.0        # c1_bn stage: input 3S^2 + in 8S^2 + out 8S^2
    return 4.0 * (WEIGHT_ELEMS + B * peak_activation_units * S ** 2)


def latency(image_size, batch, theta):
    """Wall-clock time of one forward pass (seconds).

    theta = {"P": effective FP32 flops/s, "BW": effective bytes/s,
             "t_launch": effective s/kernel}
    T = sum_ops max(flops_op/P, bytes_op/BW) + N_KERNELS * t_launch

    P, BW, t_launch are fitted effective parameters (see module docstring):
    P correlates with BW, and only the product N_KERNELS * t_launch is
    identified (N_KERNELS = 23 is an assumption, one launch per op).
    """
    S, B = _as(image_size), _as(batch)
    P, BW, t_launch = float(theta["P"]), float(theta["BW"]), float(theta["t_launch"])
    total = np.zeros(np.broadcast(S, B).shape)
    for k, (name, iu, ic, ou, oc, w, nk, cin_k2) in enumerate(_OPS):
        f_op = B * (_FU[k] * S ** 2 + _FC[k])                   # FLOPs of this op
        b_op = 4.0 * (B * (iu * S ** 2 + ic + ou * S ** 2 + oc) + w)  # bytes
        total = total + np.maximum(f_op / P, b_op / BW)
    return total + N_KERNELS * t_launch


def energy(image_size, batch, theta_energy):
    """Energy of one forward pass (joules).

    theta_energy = {"c0": J (regression intercept, effective),
                    "c_f": J/flop, "c_b": J/byte (effective; c_b may be < 0)}
    E = c0 + c_f * FLOPs + c_b * BytesMoved

    The coefficients are regression parameters, not physical energy costs:
    FLOPs and BytesMoved are collinear on the measurement grid (r ~ 1.0),
    so only their combination is identified.
    """
    c0 = float(theta_energy["c0"])
    c_f = float(theta_energy["c_f"])
    c_b = float(theta_energy["c_b"])
    return c0 + c_f * flops(image_size, batch) + c_b * bytes_moved(image_size, batch)


# ----------------------------------------------------------------------------
# self-check: compare the equations against a naive layer-by-layer computation
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    S = 224
    # brute force over the architecture
    ch = [3, 32, 32, 64, 128, 256, 256, 512]
    res = [S, S // 2, S // 4, S // 4, S // 8, S // 8, S // 16, S // 16]
    ks = [7, 0, 5, 3, 1, 3, 1]  # 0 = pool
    f = 0.0
    # convs
    for i, (k, ci, co, r) in enumerate(zip(ks, ch[:-1], ch[1:], res[1:])):
        if k == 0:
            f += 0.0                                    # pool comparisons
            continue
        f += 2 * (r * r) * co * ci * k * k              # conv MACs*2
        f += 2 * (r * r) * co + 1 * (r * r) * co        # bn + relu per output elem
    r16 = S // 16
    f += (r16 * r16) * 512                            # gap: (n-1 adds + 1 div)/channel = 2S^2
    f += 2 * 512 * 256 + 256 + 2 * 256 * 100            # fc1+relu+fc2
    print("brute FLOPs(S=224,B=1):", f, " equation:", flops(224, 1))
    assert abs(f - flops(224, 1)) < 1e-6 * f
    print("weights elems:", WEIGHT_ELEMS, "traffic u/c:", _TRAFFIC_U, _TRAFFIC_C,
          "N kernels:", N_KERNELS)
    print("flops(32..512, 1..256) broadcast ok:",
          flops(np.array([[32], [64]]), np.array([1, 256])).shape)
