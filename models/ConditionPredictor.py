import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class FeatureProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            Transpose(1, 2),  # [B, T, D] -> [B, D, T]
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Transpose(1, 2),  # [B, D, T] -> [B, T, D]
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            Transpose(1, 2),  # [B, T, D] -> [B, D, T]
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Transpose(1, 2),  # [B, D, T] -> [B, T, D]
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        return self.net(x)


class SubPredictor(nn.Module):
    def __init__(self, input_dim, out_dim, use_sigmoid=False):
        super().__init__()

        self.linear = nn.Linear(input_dim, out_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        x = self.linear(x)

        if self.use_sigmoid:
            return torch.sigmoid(x)
        return x
    
class ConditionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.feature = FeatureProcessor(input_dim, hidden_dim)

        self.f0_predictor = SubPredictor(hidden_dim, 1)
        self.energy_predictor = SubPredictor(hidden_dim, 1, use_sigmoid=True)
        self.vuv_predictor = SubPredictor(hidden_dim, 1, use_sigmoid=True)

    def forward(self, x):
        # x = [B, T, D]
        x = self.feature(x)

        f0 = self.f0_predictor(x)
        energy = self.energy_predictor(x)
        vuv = self.vuv_predictor(x)
        return f0, energy, vuv
    
# test
if __name__ == "__main__":
    batch_size = 2
    seq_len = 50
    input_dim = 16
    hidden_dim = 128

    model = ConditionPredictor(input_dim, hidden_dim)

    x = torch.randn(batch_size, seq_len, input_dim)

    f0, energy, vuv = model(x)

    # [B, T, 1] 으로 나와야 정상 동작
    print(f"f0.shape : {f0.shape}")
    print(f"energy.shape : ", energy.shape)
    print(f"vuv.shape : ", vuv.shape)

    # segmoid가 되었기 때문에 0 ~ 1 사이의 값이여야 정상 동작
    print (f"energy sigmoid | min : {energy.min().item()} / max : {energy.max().item()}")
    print (f"vuv sigmoid | min : {vuv.min().item()} / max : {vuv.max().item()}")