import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, dropout = 0.2):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.bn(out)
        out = self.dropout(out)

        return x + out

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, dropout=0.2, num_blocks=3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_dim, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        self.final_linear = nn.Linear(hidden_dim, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : [B, T, input_dim]
        x = self.input_proj(x) # [B, T, hidden_dim]
        x = x.transpose(1, 2) # [B, hidden_dim, T]
        
        for block in self.res_blocks: x = block(x)

        x = x.transpose(1, 2) # [B, T, hidden_dim]
        x = self.final_linear(x) # [B, T, 1]
        x = self.final_sigmoid(x) # [B, T, 1]

        return x.squeeze(-1) # [B, T]
    
# test
if __name__ == "__main__":
    batch = 4
    seq_len = 100
    input_dim = 80

    model = Discriminator(input_dim=input_dim)

    x = torch.randn(batch, seq_len, input_dim)
    y = model(x)

    print(f"output shape : {y.shape}")