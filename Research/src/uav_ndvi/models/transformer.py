"""Adapted from ArCo Group AgriMatNetQuantile (MIT); see ../licenses/AgriMatNet-LICENSE.

Preserves dual encoders, pre-norm ReLU blocks, masked history mean, and fusion head.
Replaces index positions with continuous elapsed days; future token contains date only.
"""
import math
import torch
from torch import nn


class ElapsedDayEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.register_buffer('frequency', torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.)/d_model)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, days):
        phase = days.unsqueeze(-1) * self.frequency
        pe = torch.stack([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        return self.dropout(x+pe)


class AgriMatNetUAV(nn.Module):
    def __init__(self, cfg, n_features=1):
        super().__init__()
        d = cfg['d_model']
        self.quantiles = cfg['quantiles']
        self.residual_prediction = cfg.get('residual_prediction', False)
        self.history_proj = nn.Linear(2*n_features+2, d)
        self.future_proj = nn.Linear(3, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg['num_heads'],
            dim_feedforward=cfg['dim_feedforward'], dropout=cfg['dropout'],
            batch_first=True, norm_first=True, activation='relu')
        self.history_encoder = nn.TransformerEncoder(layer, cfg['num_layers'], enable_nested_tensor=False)
        self.future_encoder = nn.TransformerEncoder(layer, cfg['num_layers'], enable_nested_tensor=False)
        self.history_pos = ElapsedDayEncoding(d, cfg['dropout'])
        self.future_pos = ElapsedDayEncoding(d, cfg['dropout'])
        self.head = nn.Sequential(nn.Dropout(cfg['dropout']), nn.Linear(d*2, d),
            nn.ReLU(), nn.Dropout(cfg['dropout']), nn.Linear(d, len(self.quantiles)))
        if self.residual_prediction:
            # Begin at persistence rather than an arbitrary NDVI offset.
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        mask = batch['pad'] | batch['missing']
        if mask.all(dim=1).any():
            raise ValueError('Every example needs at least one valid historical NDVI')
        h = self.history_pos(self.history_proj(batch['history']), batch['days'])
        h = self.history_encoder(h, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1)
        summary = h.masked_fill(~keep, 0.).sum(1)/keep.sum(1).clamp(min=1)
        f = self.future_proj(batch['future']).unsqueeze(1)
        f = self.future_pos(f, batch['target_day'][:, None])
        f = self.future_encoder(f)[:, 0]
        result = self.head(torch.cat([f, summary], dim=-1))
        if self.residual_prediction:
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = positions.masked_fill(mask, -1).max(dim=1).values
            latest = batch['history'][torch.arange(len(mask), device=mask.device), last, 0]
            result = result + latest[:, None]
        return result


def quantile_loss(pred, target, quantiles):
    # Vectorized original pinball loss; dataset guarantees finite vegetation targets.
    q = pred.new_tensor(quantiles)
    error = target[:, None]-pred
    return torch.maximum(q*error, (q-1)*error).mean()


__all__ = ['AgriMatNetUAV', 'ElapsedDayEncoding', 'quantile_loss']
