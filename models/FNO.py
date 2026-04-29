import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# 1) Fourier Layer (Spectral Convolution)
######################################################################

class SpectralConv2d(nn.Module):
    """
    Original FNO 2D spectral convolution layer.
    Performs FFT → truncation to low-frequency modes → linear transform → iFFT.
    """
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Complex weights for Fourier modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y))

    def compl_mul2d(self, input, weights_real, weights_imag):
        # (B, C_in, H, W) · (C_in, C_out, H, W) → (B, C_out, H, W)
        real = torch.einsum("bchw,cohw->bohw", input.real, weights_real) - \
               torch.einsum("bchw,cohw->bohw", input.imag, weights_imag)

        imag = torch.einsum("bchw,cohw->bohw", input.real, weights_imag) + \
               torch.einsum("bchw,cohw->bohw", input.imag, weights_real)

        return torch.complex(real, imag)

    def forward(self, x):
        B, C, H, W = x.shape

        # FFT2
        x_ft = torch.fft.rfftn(x, dim=[2,3])

        # Initialize output Fourier tensor
        out_ft = torch.zeros(B, self.out_channels, H, W//2+1, dtype=torch.cfloat, device=x.device)

        # Apply spectral weights in low-frequency modes
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes_x, :self.modes_y], 
                self.weights_real, 
                self.weights_imag
            )

        # iFFT2
        x = torch.fft.irfftn(out_ft, s=(H, W), dim=[2,3])
        return x


######################################################################
# 2) FNO Block
######################################################################

class FNOBlock(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()

        self.spectral_conv = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise_conv = nn.Conv2d(width, width, kernel_size=1)
        self.bn = nn.BatchNorm2d(width)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        x = x1 + x2
        x = self.bn(x)
        return F.gelu(x)


######################################################################
# 3) FNO 2D: backbone
######################################################################

class FNO(nn.Module):
    """
    Full Fourier Neural Operator for 2D PDE surrogate modeling.
    input  : [B, 19, 596, 596]
    output : [B,  3, 596, 596]
    """
    def __init__(self, in_channels=19, out_channels=3, modes=16, width=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width

        # Lift input to "width" channels
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        # 4 FNO layers (standard)
        self.block1 = FNOBlock(width, modes, modes)
        self.block2 = FNOBlock(width, modes, modes)
        self.block3 = FNOBlock(width, modes, modes)
        self.block4 = FNOBlock(width, modes, modes)

        # Output projection
        self.fc1 = nn.Conv2d(width, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        # Lift
        x = self.fc0(x)   # [B, width, H, W]

        # FNO layers
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Projection
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # final output [B,3,H,W]
        x[:, 0:1, :, :] = F.relu(x[:, 0:1, :, :])

        return x
