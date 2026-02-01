"""
ConvNeXt-V2-Style Lightweight CNN for CIFAR-10 Classification
Designed to stay under 50 MFLOPs while achieving ~90% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ NORMALIZATION ------------------

class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D features."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


# ------------------ DROP PATH ------------------

class DropPath(nn.Module):
    """Stochastic Depth (DropPath)"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ------------------ GRN ------------------

class GRN(nn.Module):
    """Global Response Normalization"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ------------------ CONVNEXT BLOCK ------------------

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-V2-style block adapted for CIFAR-10.
    """

    def __init__(self, dim, expansion_ratio=4, layer_scale_init=1e-3, drop_path=0.0):
        super().__init__()

        hidden_dim = int(dim * expansion_ratio)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + self.drop_path(x)


# ------------------ DOWNSAMPLE ------------------

class Downsample(nn.Module):
    """Downsampling using LayerNorm + strided conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


# ------------------ MODEL ------------------

class ConvNeXtV2Tiny(nn.Module):
    """
    Lightweight ConvNeXt-V2-style network for CIFAR-10.
    """

    def __init__(self, num_classes=10, dims=[32, 64, 128], depths=[2, 2, 2]):
        super().__init__()

        self.dims = dims
        self.depths = depths

        # -------- STEM (LayerNorm instead of BatchNorm) --------
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, padding=1),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        # -------- STAGES --------
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, 0.2, total_blocks)

        idx = 0
        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dims[i],
                        expansion_ratio=4,
                        layer_scale_init=1e-3,
                        drop_path=dp_rates[idx].item()
                    )
                )
                idx += 1
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1:
                self.downsamples.append(Downsample(dims[i], dims[i + 1]))

        # -------- HEAD --------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            LayerNorm1d(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.head(x)
        return x


# ------------------ FLOPs COUNT ------------------

def count_flops_params(model, input_size=(1, 3, 32, 32), device='cuda'):
    from thop import profile, clever_format

    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs

    flops_f, params_f = clever_format([flops, params], "%.3f")
    macs_f = clever_format([macs], "%.3f")

    print("=" * 50)
    print("Model Complexity Analysis")
    print("=" * 50)
    print(f"MACs: {macs_f}")
    print(f"FLOPs: {flops_f}")
    print(f"Parameters: {params_f}")
    print("=" * 50)

    return flops, params, macs


# ------------------ MAIN ------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtV2Tiny()
    model = model.to(device)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    count_flops_params(model, device=device)
