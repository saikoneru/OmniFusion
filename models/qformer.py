import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeLayerNorm(nn.LayerNorm):
    """
    LayerNorm variant that avoids NaNs by clamping small variance
    and replacing NaNs/Infs in outputs.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, eps=1e-5
        )
        # replace NaNs/Infs if any
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out


class CrossAttnBlock(nn.Module):
    """Queries attend to key/value tokens (cross-attention)."""
    def __init__(self, embed_dim, num_heads, dim_feedforward=None, dropout=0.0, activation="gelu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward or (2 * embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = SafeLayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = SafeLayerNorm(embed_dim)

    def forward(self, queries, kv, key_padding_mask=None):
        attn_out, _ = self.cross_attn(queries, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        queries = self.norm1(queries + self.dropout1(attn_out))
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        return queries


class WindowLevelQFormer(nn.Module):
    """Stable window-level Q-Former for speech embeddings (cross-attention only)."""
    def __init__(self,
                 speech_dim,
                 embed_dim=768,
                 num_queries=4,
                 num_heads=8,
                 num_layers=2,
                 dim_feedforward=None,
                 window_size=50,
                 dropout=0.0,
                 activation="gelu",
                 init_std=0.02):
        super().__init__()
        self.window_size = window_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.speech_dim = speech_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.init_std = init_std
        self.dim_feedforward = dim_feedforward
        self.input_ln = SafeLayerNorm(self.embed_dim)
        self.output_ln = SafeLayerNorm(self.embed_dim)
        self.query_tokens = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=self.init_std)
        self.input_proj = nn.Linear(self.speech_dim, self.embed_dim) if self.speech_dim != self.embed_dim else nn.Identity()
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim=self.embed_dim,
                           num_heads=self.num_heads,
                           dim_feedforward=self.dim_feedforward,
                           dropout=self.dropout,
                           activation=self.activation)
            for _ in range(self.num_layers)
        ])

        # Input projection + LayerNorm

        # Learnable queries (shared across windows)

        # Cross-attention blocks


    def post_init(self):
        self.input_ln = SafeLayerNorm(self.embed_dim)
        self.output_ln = SafeLayerNorm(self.embed_dim)
        self.query_tokens = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=self.init_std)
        self.input_proj = nn.Linear(self.speech_dim, self.embed_dim) if self.speech_dim != self.embed_dim else nn.Identity()
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim=self.embed_dim,
                           num_heads=self.num_heads,
                           dim_feedforward=self.dim_feedforward,
                           dropout=self.dropout,
                           activation=self.activation)
            for _ in range(self.num_layers)
        ])

    def forward(self, inputs, padding_mask=None, return_mask=True):
        """
        inputs: [B, T, speech_dim]
        padding_mask: [B, T] 1=valid, 0=pad or bool mask (True=valid)
        returns: [B, num_windows * num_queries, embed_dim]
        """
        B, T, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        x = self.input_proj(inputs)
        x = self.input_ln(x)

        if padding_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            if padding_mask.dtype != torch.bool:
                valid_mask = padding_mask.to(dtype=torch.long) != 0
            else:
                valid_mask = padding_mask

        # pad sequence so divisible by window_size
        win = self.window_size
        pad_len = (win - (T % win)) % win
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, self.embed_dim, device=device, dtype=dtype)], dim=1)
            valid_mask = torch.cat([valid_mask, torch.zeros(B, pad_len, dtype=torch.bool, device=device)], dim=1)

        total_T = x.size(1)
        num_windows = total_T // win
        x_windows = x.view(B, num_windows, win, self.embed_dim)
        mask_windows = valid_mask.view(B, num_windows, win)

        # Prepare queries
        qtokens = self.query_tokens.unsqueeze(0).unsqueeze(0).expand(B, num_windows, -1, -1).contiguous()
        qtokens = qtokens.view(B * num_windows, self.num_queries, self.embed_dim)

        # KV tokens
        kv = x_windows.view(B * num_windows, win, self.embed_dim)

        # key_padding_mask: True = masked
        key_padding_mask = (~mask_windows).view(B * num_windows, win)

        # Cross-attention stack
        queries = qtokens
        for layer in self.layers:
            queries = layer(queries, kv, key_padding_mask=key_padding_mask)

        queries = self.output_ln(queries)
        queries = queries.view(B, num_windows * self.num_queries, self.embed_dim)

        if return_mask:
            # compute which windows are "all padded"
            window_valid = mask_windows.any(dim=-1)    # [B, num_windows], True if any frame valid
            query_valid_mask = window_valid.unsqueeze(-1).expand(-1, -1, self.num_queries)  # [B, num_windows, num_queries]
            query_valid_mask = query_valid_mask.reshape(B, num_windows * self.num_queries)  # flatten
            return queries, query_valid_mask


        if not torch.isfinite(queries).all():
            raise RuntimeError("NaN/Inf detected in Q-Former outputs.")

        return queries

