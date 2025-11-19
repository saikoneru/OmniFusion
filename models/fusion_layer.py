import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, mode="attention"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode.lower()

        if self.mode == "concat":
            self.proj = nn.Linear(3 * hidden_dim, hidden_dim)
        elif self.mode == "attention":
            self.query = nn.Linear(hidden_dim, hidden_dim)
        elif self.mode == "gated":
            self.gate_layer = nn.Linear(3 * hidden_dim, 3)
        elif self.mode == "weighted":
            self.alpha = nn.Parameter(torch.randn(3))
        elif self.mode == "mid":
            pass
        elif self.mode == "last":
            pass
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, h1, h2, h3):
        """
        h1, h2, h3: [B, L, D]
        returns: fused [B, L, D]
        """

        if self.mode == "mid":
            z = h2
        if self.mode == "last":
            z = h3

        if self.mode == "weighted":
            weights = F.softmax(self.alpha, dim=0)  # [3]
            z = weights[0] * h1 + weights[1] * h2 + weights[2] * h3

        elif self.mode == "concat":
            z = torch.cat([h1, h2, h3], dim=-1)  # [B, L, 3D]
            z = self.proj(z)  # [B, L, D]

        elif self.mode == "attention":
            # Stack into [B, L, K, D], where K=3
            H = torch.stack([h1, h2, h3], dim=2)

            # Query from the average state [B, L, D]
            q = self.query(H.mean(dim=2))  # [B, L, D]

            # Compute attention scores: dot(q, h_k) for each state
            attn_scores = torch.sum(q.unsqueeze(2) * H, dim=-1)  # [B, L, K]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, K]

            # Weighted sum over K states
            z = (attn_weights.unsqueeze(-1) * H).sum(dim=2)  # [B, L, D]

        elif self.mode == "gated":
            concat = torch.cat([h1, h2, h3], dim=-1)  # [B, L, 3D]
            gate_logits = self.gate_layer(concat)  # [B, L, 3]
            gates = F.softmax(gate_logits, dim=-1)
            z = (gates[..., 0].unsqueeze(-1) * h1 +
                 gates[..., 1].unsqueeze(-1) * h2 +
                 gates[..., 2].unsqueeze(-1) * h3)

        return z
