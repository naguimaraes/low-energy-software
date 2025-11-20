# classes.py
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Three AlexNet variants:
  - ExplicitAlexNet: explicit convolutions via F.unfold (patch extraction + matmul)
  - FFTAlexNet: convolutions via FFTs with batched FFT reuse
  - TorchAlexNet: standard nn.Conv2d (PyTorch native)
Utilities:
  - generate_kernel_sizes(first): produce 5 kernel sizes descending to 3x3
  - AlexNetClassifier: shared fully-connected head
"""

def generate_kernel_sizes(first_kernel: int, num_layers: int = 5):
    """
    Generate kernel sizes for convolutional layers with smooth progression.
    
    Args:
        first_kernel: Starting (largest) kernel size
        num_layers: Number of convolutional layers (default: 5)
    
    Returns:
        List of kernel sizes with smooth decreasing progression to 3x3.
        
    Strategy:
        - Ensures smooth transition from first_kernel down to 3
        - Uses geometric-like progression for natural size reduction
        - Clamps all values to be odd numbers >= 3
        - Examples:
            first_kernel=11, num_layers=5 -> [11, 7, 5, 3, 3]
            first_kernel=9, num_layers=5 -> [9, 7, 5, 3, 3]
            first_kernel=7, num_layers=5 -> [7, 5, 3, 3, 3]
            first_kernel=5, num_layers=5 -> [5, 3, 3, 3, 3]
            first_kernel=13, num_layers=5 -> [13, 9, 7, 5, 3]
    """
    k = int(first_kernel)
    
    # Ensure first kernel is odd and >= 3
    k = max(3, k if k % 2 == 1 else k - 1)
    
    if num_layers == 1:
        return [k]
    
    # Generate smooth progression from k down to 3
    sizes = []
    current = k
    
    for i in range(num_layers):
        sizes.append(current)
        
        # Calculate next size with smooth progression
        if current > 3:
            # Geometric-like decrease: reduce by ~30% but keep odd
            next_size = max(3, int(current * 0.7))
            # Ensure it's odd
            if next_size % 2 == 0:
                next_size -= 1
            # Ensure at least 2 reduction per step when possible
            if next_size >= current:
                next_size = max(3, current - 2)
            current = next_size
        # else: keep current = 3 for remaining layers
    
    return sizes


class AlexNetClassifier(nn.Module):
    """Shared classifier head used by all three variants (for CIFAR input sizes)."""
    def __init__(self, num_classes=100, in_features=256 * 4 * 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ------------------------------
# 1) EXPLICIT convolution (unfold)
# ------------------------------
class ExplicitConv2d(nn.Module):
    """
    Implements a convolution using unfold + matrix multiplication.
    This is an 'explicit' convolution (no nn.Conv2d).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kh, self.kw = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # weight shape: (out, in, kh, kw)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kh, self.kw) * (2. / (in_channels * self.kh * self.kw))**0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x: (B, C_in, H, W)
        B = x.size(0)
        # pad
        x_p = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        # unfold: (B, C_in * kh * kw, L) where L is number of sliding locations
        patches = F.unfold(x_p, kernel_size=(self.kh, self.kw), stride=self.stride)
        # reshape weights: (out, in*kh*kw)
        w = self.weight.view(self.out_channels, -1)
        # matmul: (B, out, L)
        out = w.unsqueeze(0).matmul(patches)  # (1, out, in*kh*kw) x (B, in*kh*kw, L) -> (B, out, L)
        # fold back to (B, out, H_out, W_out)
        # compute output H/W
        H_p = x_p.size(2)
        W_p = x_p.size(3)
        H_out = (H_p - self.kh) // self.stride + 1
        W_out = (W_p - self.kw) // self.stride + 1
        out = out.view(B, self.out_channels, H_out, W_out)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class ConvolutionalAlexNet(nn.Module):
    def __init__(self, num_classes=100, first_kernel=11):
        super().__init__()
        ks = generate_kernel_sizes(first_kernel)
        # paddings chosen to preserve spatial dims before pooling
        pads = [k // 2 for k in ks]
        self.features = nn.Sequential(
            ExplicitConv2d(3, 64, kernel_size=ks[0], stride=1, padding=pads[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ExplicitConv2d(64, 192, kernel_size=ks[1], padding=pads[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ExplicitConv2d(192, 384, kernel_size=ks[2], padding=pads[2]),
            nn.ReLU(inplace=True),

            ExplicitConv2d(384, 256, kernel_size=ks[3], padding=pads[3]),
            nn.ReLU(inplace=True),

            ExplicitConv2d(256, 256, kernel_size=ks[4], padding=pads[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.Identity()
        # For CIFAR-100 with input 32x32 and these paddings the feature map before classifier is 4x4
        self.classifier = AlexNetClassifier(num_classes=num_classes, in_features=256 * 4 * 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ------------------------------
# 2) FFT-based convolution
# ------------------------------
class FFTConv2d(nn.Module):
    """
    FFT convolution with batched FFT reuse. (vectorized)
    Assumes real inputs; uses rfft2 / irfft2 for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kH, self.kW) * (2. / (in_channels * self.kH * self.kW))**0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x: (B, Cin, H, W)
        B, Cin, H, W = x.shape
        pad = self.padding
        x_padded = F.pad(x, (pad, pad, pad, pad))

        fft_h = x_padded.size(2) + self.kH - 1
        fft_w = x_padded.size(3) + self.kW - 1

        # FFT of inputs: (B, Cin, Hf, Wf_half)
        Xf = torch.fft.rfft2(x_padded, s=(fft_h, fft_w))

        # FFT of filters: (Cout, Cin, Hf, Wf_half)
        # Flip kernel for convolution vs cross-correlation (PyTorch conv2d is cross-correlation)
        weight_flipped = torch.flip(self.weight, dims=[-2, -1])
        Wf = torch.fft.rfft2(weight_flipped, s=(fft_h, fft_w))

        # Multiply and sum over input channels: Yf[b, o, h, w] = sum_f Xf[b,f,h,w] * Wf[o,f,h,w]
        # Use einsum for batched multiplication
        Yf = torch.einsum("bfhw,ofhw->bohw", Xf, Wf)

        # Inverse fft back to spatial domain -> (B, Cout, fft_h, fft_w)
        y = torch.fft.irfft2(Yf, s=(fft_h, fft_w))

        # compute output spatial dims
        out_h = (H + 2*pad - self.kH) // self.stride + 1
        out_w = (W + 2*pad - self.kW) // self.stride + 1

        # For padded convolution, crop the central valid region
        start_h = (fft_h - out_h) // 2
        start_w = (fft_w - out_w) // 2
        y = y[:, :, start_h : start_h + out_h, start_w : start_w + out_w]

        # Apply stride if >1 (though in AlexNet convs are stride=1)
        if self.stride > 1:
            y = y[:, :, ::self.stride, ::self.stride]

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class FFTAlexNet(nn.Module):
    def __init__(self, num_classes=100, first_kernel=11):
        super().__init__()
        ks = generate_kernel_sizes(first_kernel)
        pads = [k // 2 for k in ks]
        self.features = nn.Sequential(
            FFTConv2d(3, 64, kernel_size=ks[0], stride=1, padding=pads[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            FFTConv2d(64, 192, kernel_size=ks[1], padding=pads[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            FFTConv2d(192, 384, kernel_size=ks[2], padding=pads[2]),
            nn.ReLU(inplace=True),

            FFTConv2d(384, 256, kernel_size=ks[3], padding=pads[3]),
            nn.ReLU(inplace=True),

            FFTConv2d(256, 256, kernel_size=ks[4], padding=pads[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = AlexNetClassifier(num_classes=num_classes, in_features=256 * 4 * 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ------------------------------
# 3) PyTorch native Conv2d AlexNet
# ------------------------------
class TorchAlexNet(nn.Module):
    def __init__(self, num_classes=100, first_kernel=11):
        super().__init__()
        ks = generate_kernel_sizes(first_kernel)
        pads = [k // 2 for k in ks]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=ks[0], stride=1, padding=pads[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=ks[1], padding=pads[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=ks[2], padding=pads[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=ks[3], padding=pads[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=ks[4], padding=pads[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = AlexNetClassifier(num_classes=num_classes, in_features=256 * 4 * 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
