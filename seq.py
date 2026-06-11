import torch
import torch.nn as nn
import math


# ── Helpers ───────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        encoded = self.dropout(x + self.pe[:, :x.size(1)])
        return encoded

class TemporalEncoding(nn.Module):
    """
    Continuous-time positional encoding — unlike the standard version which
    assumes fixed integer steps, this uses actual timestamps so irregular
    and variable sampling rates are handled correctly.

    Args:
        d_model: embedding dimension
        dropout: dropout rate

    Forward:
        x: (B, T, d_model)
        t: (B, T) — actual timestamps (any unit: seconds, ms, etc.)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Frequency bands — equivalent to div_term in standard PE
        # but applied dynamically to continuous t rather than integer positions
        # Shape: (d_model // 2,) — one frequency per sin/cos pair
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div', div)  # (d_model // 2,)

    def forward(self, x, t):
        """
        x: (B, T, d_model)
        t: (B, T)            — timestamps, can be irregular
        """
        # t: (B, T) → (B, T, 1) to broadcast over frequency bands
        t = t.unsqueeze(-1)                          # (B, T, 1)
        args = t/2 * self.div                          # (B, T, d_model // 2)

        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)

        return self.dropout(x + pe)

class ContinuousRoPE(nn.Module):
    # cruisin' for a bruisin'
    def __init__(self, d_head, max_t=2000.0):
        super().__init__()
        self.max_t = max_t
        # Frequencies: k-th band completes (k+1) cycles over max_t
        # so bands range from 1 cycle (lowest) to d_head//2 cycles (highest)
        k = torch.arange(1, d_head // 2 + 1)
        freqs = 2 * math.pi * k / max_t
        self.register_buffer('freqs', freqs)  # (d_head // 2,)

    def _rotate(self, x, t):
        angles = t.unsqueeze(-1) * self.freqs  # (B, T, d_head//2)
        sin    = torch.sin(angles).unsqueeze(2)
        cos    = torch.cos(angles).unsqueeze(2)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], dim=-1).flatten(-2)
        

    def forward(self, q, k, t):
        return self._rotate(q, t), self._rotate(k, t)

def project_to_SO3(R):
    """
    x: (..., 9) → (..., 3, 3) → SVD → proper rotation matrix (..., 3, 3)
    Handles reflection by flipping the last singular vector if det < 0.
    """
    shape = R.shape[:-2]
    U, _, Vh = torch.linalg.svd(R)
    # Fix reflections: ensure det(U @ Vh) = +1
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, device=R.device).unsqueeze(0).expand(*shape, -1, -1).clone()
    D[..., 2, 2] = d.sign()
    return U @ D @ Vh  # (..., 3, 3)

def shift_frame_sequence(rot):
    start  = torch.eye(3, device=rot.device).flatten().expand(1, -1)  # (B, 1, 9)
    rot_flat = rot.permute(0,2,1).flatten(-2)
    rot_shifted = torch.cat([start, rot_flat[:-1]], dim=0)
    return rot_shifted

# ── Set Encoder ───────────────────────────────────────────────────────────────

class SetAttentionBlock(nn.Module):
    """
    One block of multi-head self-attention over an unordered set.
    No positional encoding — order invariant by design.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # padding_mask: (B, N) — passed as key_padding_mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

class SetEncoder(nn.Module):
    """
    Encodes 3 sets of N noisy 3D measurements into a single context vector.

    Input:  e1, e2, e3 each (B, N, 3)
    Output: memory (B, 3, d_model)  — one token per basis direction,
            which the decoder can cross-attend to
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            SetAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Learned pooling: collapses N measurements → 1 summary token
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
    def encode_one(self, x, padding_mask=None):
        """
        x:            (B, N, 3)
        padding_mask: (B, N) — True where padded (i.e. not real measurements)
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        q = self.pool_query.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        # Key/value padding mask for attention pooling
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=padding_mask)
        return pooled

    def forward(self, e1, e2, e3, mask_e1=None, mask_e2=None, mask_e3=None):
        t1 = self.encode_one(e1, mask_e1)
        t2 = self.encode_one(e2, mask_e2)
        t3 = self.encode_one(e3, mask_e3)
        return torch.cat([t1, t2, t3], dim=1)

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead  = nhead
        self.d_head = d_model // nhead
        self.scale  = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.rope    = ContinuousRoPE(self.d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t, attn_mask=None, key_padding_mask=None):
        """
        x:               (B, T, d_model)
        t:               (B, T) — timestamps
        attn_mask:       (T, T) — e.g. causal mask
        key_padding_mask:(B, T) — True where padded
        """
        B, T, _ = x.shape

        def project_and_split(proj, x):
            return proj(x).reshape(B, T, self.nhead, self.d_head)

        q = project_and_split(self.q_proj, x)  # (B, T, nhead, d_head)
        k = project_and_split(self.k_proj, x)
        v = project_and_split(self.v_proj, x)

        # Apply RoPE — rotation depends on timestamp, relative by construction
        q, k = self.rope(q, k, t)

        # Standard scaled dot-product attention
        # Rearrange to (B, nhead, T, d_head) for matmul
        q = q.transpose(1, 2)   # (B, nhead, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nhead, T, T)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v)                               # (B, nhead, T, d_head)

        out  = out.transpose(1, 2).reshape(B, T, -1)               # (B, T, d_model)
        return self.o_proj(out)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn  = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, t, attn_mask=None, key_padding_mask=None):
        x = self.norm1(x + self.drop(self.attn(x, t, attn_mask, key_padding_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, t, attn_mask=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, t, attn_mask, key_padding_mask)
        return x