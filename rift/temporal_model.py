from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        if mask is None:
            return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

        mask = mask.to(dtype=x.dtype)
        front_m = mask[:, 0:1, :].repeat(1, pad, 1)
        end_m = mask[:, -1:, :].repeat(1, pad, 1)
        mask_pad = torch.cat([front_m, mask, end_m], dim=1)
        valid = 1.0 - mask_pad

        bsz, seq_len, num_nodes = x_pad.shape
        x_reshape = x_pad.permute(0, 2, 1).reshape(bsz * num_nodes, 1, seq_len)
        v_reshape = valid.permute(0, 2, 1).reshape(bsz * num_nodes, 1, seq_len)
        weight = torch.ones(1, 1, self.kernel_size, device=x.device, dtype=x.dtype)
        num = F.conv1d(x_reshape * v_reshape, weight, stride=1)
        den = F.conv1d(v_reshape, weight, stride=1)
        mean = num / (den + 1e-6)
        return mean.reshape(bsz, num_nodes, -1).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        trend = self.moving_avg(x, mask=mask)
        seasonal = x - trend
        return seasonal, trend


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=(1, 1), bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(4 * hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


class BaseBranch(nn.Module):
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        num_layers: int,
        model_dim: int,
        prompt_dim: int,
        tod_size: int,
        kernel_size: int,
        disable_series_decomposition: bool = False,
        disable_temporal_tod: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.embed_dim = model_dim
        self.prompt_dim = prompt_dim
        self.hidden_dim = self.embed_dim + 2 * self.prompt_dim
        self.time_of_day_size = tod_size
        self.day_of_week_size = 7
        self.disable_series_decomposition = bool(disable_series_decomposition)
        self.disable_temporal_tod = bool(disable_temporal_tod)

        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.prompt_dim))
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.prompt_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.decomposition = SeriesDecomposition(kernel_size)
        self.time_series_seasonal = nn.Conv1d(seq_len, self.embed_dim, kernel_size=1, bias=True)
        self.time_series_trend = nn.Conv1d(seq_len, self.embed_dim, kernel_size=1, bias=True)
        self.encoder = nn.Sequential(*[FeedForward(self.hidden_dim) for _ in range(num_layers)])
        self.prediction_head = nn.Conv2d(self.hidden_dim, horizon, kernel_size=(1, 1), bias=True)

    @staticmethod
    def _safe_time_index(data: torch.Tensor, size: int) -> torch.Tensor:
        data = torch.clamp(data, 0.0, 1.0 - 1e-6)
        idx = (data * size).long()
        return torch.clamp(idx, 0, size - 1)

    def forward(self, history_data: torch.Tensor):
        tod_idx = self._safe_time_index(history_data[:, -1, :, 1], self.time_of_day_size)
        dow_idx = self._safe_time_index(history_data[:, -1, :, 2], self.day_of_week_size)
        tod_emb = self.time_in_day_emb[tod_idx].transpose(1, 2).unsqueeze(-1)
        if self.disable_temporal_tod:
            tod_emb = torch.zeros_like(tod_emb)
        dow_emb = self.day_in_week_emb[dow_idx].transpose(1, 2).unsqueeze(-1)

        speed = history_data[..., 0]
        seasonal, trend = self.decomposition(speed)
        if self.disable_series_decomposition:
            seasonal, trend = speed, speed
        series_emb = self.time_series_seasonal(seasonal) + self.time_series_trend(trend)
        series_emb = series_emb.unsqueeze(-1)

        hidden = torch.cat([series_emb, tod_emb, dow_emb], dim=1)
        base_tokens = hidden.transpose(1, -1)
        encoded = self.encoder(hidden)
        encoded_tokens = encoded.transpose(1, -1)
        prediction = self.prediction_head(encoded)
        return base_tokens, encoded_tokens, prediction

    def residual_projection(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(hidden_tokens.transpose(1, -1))
        return hidden.transpose(1, -1)


class CompactForecastingModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        tod_size: int,
        num_layers: int = 3,
        model_dim: int = 64,
        prompt_dim: int = 32,
        kernel_size: int = 3,
        decoder_hidden_dim: int = 256,
        disable_series_decomposition: bool = False,
        disable_temporal_tod: bool = False,
        disable_base_prediction: bool = False,
        disable_temporal_residual: bool = False,
    ) -> None:
        super().__init__()
        self.base = BaseBranch(
            seq_len=seq_len,
            horizon=horizon,
            num_layers=num_layers,
            model_dim=model_dim,
            prompt_dim=prompt_dim,
            tod_size=tod_size,
            kernel_size=kernel_size,
            disable_series_decomposition=disable_series_decomposition,
            disable_temporal_tod=disable_temporal_tod,
        )
        self.disable_base_prediction = bool(disable_base_prediction)
        self.disable_temporal_residual = bool(disable_temporal_residual)
        self.decoder = nn.Sequential(
            nn.Linear(self.base.hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, horizon),
        )

    def forward(self, history_data: torch.Tensor, return_parts: bool = False):
        base_tokens, _encoded_tokens, base_prediction = self.base(history_data)
        if self.disable_base_prediction:
            base_prediction = torch.zeros_like(base_prediction)
        if self.disable_temporal_residual:
            residual_prediction = torch.zeros_like(base_prediction)
        else:
            residual_tokens = self.base.residual_projection(base_tokens)
            residual_prediction = self.decoder(residual_tokens).transpose(1, -1)
        if return_parts:
            return base_prediction, residual_prediction
        return base_prediction + residual_prediction
