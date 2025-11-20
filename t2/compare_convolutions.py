import torch
import torch.nn as nn
import torch.nn.functional as F
from classes import FFTConv2d, ExplicitConv2d

def test_fft_vs_standard():
    # Test parameters
    batch_size = 1
    in_channels = 3
    out_channels = 64
    kernel_size = 5
    stride = 1
    padding = 2  # same padding
    height, width = 32, 32

    # Create input
    x = torch.randn(batch_size, in_channels, height, width)

    # Create standard conv
    conv_standard = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    # Create FFT conv with same weights
    conv_fft = FFTConv2d(in_channels, out_channels, kernel_size, stride, padding)
    conv_fft.weight.data = conv_standard.weight.data.clone()
    conv_fft.bias.data = conv_standard.bias.data.clone()

    # Create explicit conv with same weights
    conv_explicit = ExplicitConv2d(in_channels, out_channels, kernel_size, stride, padding)
    conv_explicit.weight.data = conv_standard.weight.data.clone()
    conv_explicit.bias.data = conv_standard.bias.data.clone()

    # Forward pass
    with torch.no_grad():
        out_standard = conv_standard(x)
        out_fft = conv_fft(x)
        out_explicit = conv_explicit(x)

    print(f"Input shape: {x.shape}")
    print(f"Standard output shape: {out_standard.shape}")
    print(f"FFT output shape: {out_fft.shape}")
    print(f"Explicit output shape: {out_explicit.shape}")

    # Check if shapes match
    shapes_match = out_standard.shape == out_fft.shape == out_explicit.shape
    print(f"Output shapes match: {shapes_match}")

    if shapes_match:
        # Check numerical equivalence
        diff_fft = torch.abs(out_standard - out_fft).max().item()
        diff_explicit = torch.abs(out_standard - out_explicit).max().item()

        print(f"Max difference FFT vs Standard: {diff_fft}")
        print(f"Max difference Explicit vs Standard: {diff_explicit}")

        # Tolerance for floating point
        tolerance = 1e-5
        fft_equivalent = diff_fft < tolerance
        explicit_equivalent = diff_explicit < tolerance

        print(f"FFT equivalent to Standard: {fft_equivalent}")
        print(f"Explicit equivalent to Standard: {explicit_equivalent}")

        if not fft_equivalent:
            print("FFT implementation has issues!")
            # Show some values
            print(f"Standard[0,0,:3,:3]:\n{out_standard[0,0,:3,:3]}")
            print(f"FFT[0,0,:3,:3]:\n{out_fft[0,0,:3,:3]}")
        else:
            print("FFT implementation is correct!")

if __name__ == "__main__":
    test_fft_vs_standard()
