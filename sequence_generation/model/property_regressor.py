# Adapted from https://github.com/SeulLee05/MOOD/blob/main/models/regressor.py
import torch
import torch.nn as nn

class SeqRegressor(nn.Module):
    def __init__(self, alphabet_size, hidden_dim=128, depth=4, dropout=0.1):
        super().__init__()

        # one-hot(A) -> hidden_dim
        self.embedder = nn.Embedding(alphabet_size, hidden_dim)

        # 1D conv stack (sequence backbone)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(depth)
        ])

        dim = (depth + 1) * hidden_dim
        self.sigmoid_linear = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())

        self.final_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        x: (B, L, A) one-hot (or soft one-hot)
        mask: (B, L) True/1 = valid token, False/0 = padding (optional)
        """
        xs = []
        out = self.embedder(x['seqs'])
        xs.append(out)

        h = out.transpose(1, 2)
        for conv in self.convs:
            h = torch.tanh(conv(h))
            xs.append(h.transpose(1, 2))

        feat = torch.cat(xs, dim=-1)

        gate = self.sigmoid_linear(feat) * self.tanh_linear(feat)

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (gate * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1.0))
        else:
            pooled = gate.mean(dim=1)

        pooled = torch.tanh(pooled)
        return self.final_linear(pooled)


class RegressorScoreX(nn.Module):
    def __init__(self, sde, regressor):
        super().__init__()
        self.sde = sde
        self.regressor = regressor

    def forward(self, x, t, mask=None):
        with torch.enable_grad():
            x_para = nn.Parameter(x)
            F = self.regressor(x_para, mask=mask).sum()
            F.backward()
            return x_para.grad