import torch
import torch.nn as nn

"""
┌───────────────────────────── WaveNetModule ──────────────────────────────┐
│                               Dilated Conv                               │
│                                    │                                     │
│   ┌─────────────────────WaveNetBlock (x N blocks) ───────────────────┐   │
│   │                                │                                 │   │
│   │                   ┌────────────┼───────────────┐                 │   │
│   │                   │      Dilated Conv    input_conv              │   │
│   │                   │            │       +       │                 │   │
│   │                   │            └───────┬───────┘                 │   │
│   │                   │            ┌───────┴───────┐                 │   │
│   │                   │      conv1x1_filter  conv1x1_gate            │   │
│   │                   │           tanh     x    sigmoid              │   │
│   │                   │            └───────┬───────┘                 │   │
│   │                   │            ┌───────┴───────┐                 │   │
│   │                   │      conv1x1_resual  conv1x1_skip            │   │
│   │                   │     +      │               │                 │   │
│   │                   └─────┬──────┘               │                 │   │
│   │                      output                   skip               │   │
│   │                         │                      │                 │   │
│   └─────────────────────────┼──────────────────────┼─────────────────┘   │
│                             │               +      │                     │
│                             │       ┌──────────────┘                     │
│                           output    │                                    │
│                                    tanh                                  │
│                                     │                                    │
│                                 final_conv                               │
│                                     │                                    │
│                                   output                                 │
│                              [B, Channels, T]                            │
└──────────────────────────────────────────────────────────────────────────┘

"""

class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, dilation = 1, skip_channels = None):
        super().__init__()

        self.dilated_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) // 2) * dilation,
            dilation=dilation
        )
        self.input_conv = nn.Conv1d(channels, channels, 1)

        self.conv1x1_filter = nn.Conv1d(channels, channels, 1)
        self.conv1x1_gate = nn.Conv1d(channels, channels, 1)
        self.conv1x1_residual = nn.Conv1d(channels, channels, 1)
        self.conv1x1_skip = nn.Conv1d(channels, skip_channels or channels, 1)

    def forward(self, x):
        h = self.dilated_conv(x) + self.input_conv(x)

        filter_out = self.conv1x1_filter(h)
        gate_out = self.conv1x1_gate(h)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        residual = self.conv1x1_residual(gated)
        skip = self.conv1x1_skip(gated)

        output = x + residual
        return output, skip
    
class WaveNetModule(nn.Module):
    def __init__(self, channels, kernel_size = 3, num_layers = 8, skip_channels = None):
        super().__init__()

        self.blocks = nn.ModuleList([
            WaveNetBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation= 2 ** i,
                skip_channels=skip_channels
            )
            for i in range(num_layers)
        ])
        
        self.final_conv = nn.Conv1d(skip_channels or channels, channels, 1)

    def forward(self, x):
        # x : [B, C, T]
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip if isinstance(skip_sum, torch.Tensor) else skip
        
        out = self.final_conv(torch.tanh(skip_sum))
        return out
    
# test
if __name__ == "__main__":
    batch_size = 2
    channels = 64
    skip_channels = 64
    seq_len = 128
    kernel_size = 3
    num_layers = 2

    # 테스트용 더미 입력 생성
    x = torch.randn(batch_size, channels, seq_len)
    print(f"input shape : {x.shape}")

    model = WaveNetModule(
        channels=channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        skip_channels=skip_channels
    )

    output = model(x)
    print(f"output shape : {output.shape}")

    if output.shape != (batch_size, channels, seq_len):
        print("Output shape mismatch")