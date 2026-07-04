"""
Temporal Convolutional Network (TCN) encoder for nowcasting.

Architecture:
  CausalConv1d → TCNBlock (×n_layers) → AdaptiveAvgPool1d → embedding vector.
  Causality is maintained by right-padding then slicing — no future information
  leaks into any output position.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal convolution: right-pad input so output[t] depends only on input[:t+1].
    
    WHY right-pad-then-slice instead of PyTorch padding="causal":
    nn.Conv1d has no built-in causal mode. Manual padding gives explicit control
    and is compatible with ONNX export (M6).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C_out, T)  — length preserved."""
        # Right-pad: add zeros to the end of the time axis
        x_padded = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        # Slice to original length (remove any extra samples from the right)
        return out[:, :, : x.size(2)]


class TCNBlock(nn.Module):
    """
    Two CausalConv1d + LayerNorm + Dropout + residual + ReLU.
    
    Residual connection uses a 1×1 conv when channel dims differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual projection when channel dimensions change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C_out, T)"""
        residual = self.residual(x)

        out = self.conv1(x)
        # LayerNorm expects (B, T, C) — transpose, norm, transpose back
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out + residual)
        out = self.dropout(out)

        return out


class TCNEncoder(nn.Module):
    """
    Multi-layer TCN encoder producing a fixed-size embedding per window.

    Input:  (B, T, F) — batch of time windows with F features.
    Output: (B, embed_dim) — one embedding vector per window.

    Default architecture: 4 layers with dilations [1, 2, 4, 8] giving a
    receptive field of (3-1)*2*(1+2+4+8) + 1 = 61 timesteps — comfortably
    covers the 60-step input window.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(n_layers)]
        assert len(dilations) == n_layers

        layers: list[nn.Module] = []
        in_ch = n_features
        for i, d in enumerate(dilations):
            out_ch = embed_dim
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — batch of time-series windows.
        Returns:
            (B, embed_dim) — pooled embedding.
        """
        # (B, T, F) → (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        out = self.network(x)          # (B, embed_dim, T)
        out = self.pool(out).squeeze(2)  # (B, embed_dim)
        return out
