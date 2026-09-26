"""models.py — the small CNN from Homework 1 (ITMO Efficient Models 2026).

Architecture (from the assignment table), input 3 x S x S, 100 classes:

    Conv 7x7  s2  3 -> 32  + BN + ReLU      (out: S/2)
    MaxPool 3x3 s2 p1                       (out: S/4)
    Conv 5x5      32 -> 64 + BN + ReLU      (out: S/4)
    Conv 3x3 s2   64 -> 128 + BN + ReLU     (out: S/8)
    Conv 1x1     128 -> 256 + BN + ReLU     (out: S/8)
    Conv 3x3 s2  256 -> 256 + BN + ReLU     (out: S/16)
    Conv 1x1     256 -> 512 + BN + ReLU     (out: S/16)
    GlobalAvgPool -> Linear 512->256 -> ReLU -> Linear 256->100

Conventions honoured: padding = k // 2, conv bias=False, ReLU(inplace=True).
Convs are followed by BatchNorm + ReLU (the standard Conv->BN->ReLU triplet;
bias=False on convs exists precisely because BN carries the bias).

torch.profiler.record_function tags are attached to every op so that
measure.py can dump the CUDA kernel name per layer into results/kernels.csv.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import record_function


def conv_bn_relu(cin: int, cout: int, k: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        # stage 1: conv7x7 s2 3->32 then maxpool 3x3 s2 p1  (S -> S/2 -> S/4)
        self.c1 = conv_bn_relu(3, 32, 7, stride=2)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.c2 = conv_bn_relu(32, 64, 5)
        self.c3 = conv_bn_relu(64, 128, 3, stride=2)
        self.c4 = conv_bn_relu(128, 256, 1)
        self.c5 = conv_bn_relu(256, 256, 3, stride=2)
        self.c6 = conv_bn_relu(256, 512, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

    @staticmethod
    def _run(sub: nn.Sequential, tag: str, x: torch.Tensor) -> torch.Tensor:
        with record_function(f"{tag}_conv"):
            x = sub[0](x)
        with record_function(f"{tag}_bn"):
            x = sub[1](x)
        with record_function(f"{tag}_relu"):
            x = sub[2](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._run(self.c1, "c1", x)
        with record_function("pool"):
            x = self.pool(x)
        x = self._run(self.c2, "c2", x)
        x = self._run(self.c3, "c3", x)
        x = self._run(self.c4, "c4", x)
        x = self._run(self.c5, "c5", x)
        x = self._run(self.c6, "c6", x)
        with record_function("gap"):
            x = self.gap(x)
        with record_function("gap_flat"):
            x = torch.flatten(x, 1)
        with record_function("fc1"):
            x = self.fc1(x)
        with record_function("fc1_relu"):
            x = self.relu7(x)
        with record_function("fc2"):
            x = self.fc2(x)
        return x


def build_model(device: str = "cuda") -> nn.Module:
    """Deterministic construction, eval mode, FP32, on `device`."""
    torch.manual_seed(0)
    model = SmallCNN()
    for p in model.parameters():
        p.data.normal_(0, 0.05)          # avoid degenerate BN stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
    model = model.to(device)
    model.eval()
    return model
