"""RIFT temporal predictor and target-private graph residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_model import CompactForecastingModel




class SparseGraphResidualAdapter(nn.Module):
    """Private sparse graph residual head.

    The graph adjacency and residual parameters are local to each city and are
    never uploaded during federated aggregation. The `topk` argument sparsifies
    local graph message passing only; it is not a federated source-selection
    mechanism.
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        graph_adj: torch.Tensor,
        hidden_dim: int = 32,
        num_layers: int = 1,
        gate_init: float = 0.1,
        gate_kind: str = "scalar",
        residual_scale: float = 1.0,
        start_step: int = 1,
        use_tod: bool = True,
        init_scale: float = 0.02,
        topk: int = 0,
    ) -> None:
        super().__init__()
        if graph_adj.dim() != 2 or graph_adj.shape[0] != graph_adj.shape[1]:
            raise ValueError(f"graph_adj must be square, got shape={tuple(graph_adj.shape)}")
        num_nodes = int(graph_adj.shape[0])
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, int(num_layers))
        self.use_tod = bool(use_tod)
        self.num_nodes = num_nodes
        self.graph_topk = max(0, int(topk))
        self.speed_basis = nn.Linear(seq_len, hidden_dim)
        self.tod_basis = nn.Linear(seq_len, hidden_dim) if self.use_tod else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.neigh_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.out = nn.Linear(hidden_dim, horizon)
        nn.init.normal_(self.out.weight, mean=0.0, std=float(init_scale))
        nn.init.zeros_(self.out.bias)

        if gate_kind not in {"scalar", "horizon"}:
            raise ValueError(f"unknown private graph gate_kind={gate_kind!r}")
        self.gate_kind = gate_kind
        init = min(1.0 - 1e-4, max(1e-4, float(gate_init)))
        logit = torch.logit(torch.tensor(init)).item()
        if self.gate_kind == "horizon":
            self.gate_logit = nn.Parameter(torch.full((horizon,), float(logit)))
        else:
            self.gate_logit = nn.Parameter(torch.tensor(float(logit)))
        self.residual_scale = float(residual_scale)

        mask = torch.ones(horizon, dtype=torch.float32)
        start = max(0, min(horizon, int(start_step) - 1))
        mask[:start] = 0.0
        self.register_buffer("horizon_mask", mask.view(1, 1, horizon), persistent=False)

        graph_adj = graph_adj.detach().clone().float()
        if 0 < self.graph_topk < num_nodes:
            vals, idx = torch.topk(graph_adj, k=self.graph_topk, dim=1, largest=True)
            self.register_buffer("graph_topk_idx", idx.long(), persistent=True)
            self.register_buffer("graph_topk_weight", vals.float(), persistent=True)
            self.graph_effective_edges = int(num_nodes * self.graph_topk)
        else:
            self.graph_topk = 0
            self.register_buffer("graph_adj", graph_adj, persistent=True)
            self.graph_effective_edges = int(num_nodes * num_nodes)

    def efficiency_audit(self) -> dict:
        dense_edges = int(self.num_nodes * self.num_nodes)
        return {
            "mode": "topk" if self.graph_topk > 0 else "dense",
            "num_nodes": int(self.num_nodes),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "topk": int(self.graph_topk),
            "dense_edges_per_layer": dense_edges,
            "effective_edges_per_layer": int(self.graph_effective_edges),
            "edge_reduction_ratio": float(dense_edges / max(int(self.graph_effective_edges), 1)),
        }

    def forward(self, speed_norm: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        # speed_norm/tod: [B, N, T]
        x = self.speed_basis(speed_norm)
        if self.tod_basis is not None:
            phase = self.tod_basis(tod)
            x = x + torch.sin(phase) + torch.cos(phase)
        x = self.norm(x)
        for self_layer, neigh_layer in zip(self.self_layers, self.neigh_layers):
            if self.graph_topk > 0:
                idx = self.graph_topk_idx.to(device=x.device)
                weight = self.graph_topk_weight.to(device=x.device, dtype=x.dtype)
                neigh_x = x[:, idx, :]
                neigh = (neigh_x * weight.view(1, self.num_nodes, self.graph_topk, 1)).sum(dim=2)
            else:
                adj = self.graph_adj.to(device=x.device, dtype=x.dtype)
                neigh = torch.einsum("ij,bjh->bih", adj, x)
            x = F.gelu(self_layer(x) + neigh_layer(neigh))
        gate = torch.sigmoid(self.gate_logit)
        if self.gate_kind == "horizon":
            gate = gate.view(1, 1, -1)
        residual = self.residual_scale * gate * self.out(x)
        return residual * self.horizon_mask


class RIFTModel(nn.Module):
    """Shared temporal pathway plus city-private graph residual calibration."""

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        tod_size: int,
        speed_mean: float,
        speed_std: float,
        num_nodes: int,
        model_dim: int = 24,
        prompt_dim: int = 12,
        num_layers: int = 2,
        decoder_hidden_dim: int = 96,
        share_base_blocks: str | None = None,
        use_private_graph_residual: bool = True,
        private_graph_hidden_dim: int = 8,
        private_graph_num_layers: int = 1,
        private_graph_gate_init: float = 0.1,
        private_graph_gate_kind: str = "horizon",
        private_graph_residual_scale: float = 1.0,
        private_graph_start_step: int = 1,
        private_graph_use_tod: bool = True,
        private_graph_init_scale: float = 0.02,
        private_graph_topk: int = 4,
        graph_adj: torch.Tensor | None = None,
        disable_series_decomposition: bool = False,
        disable_temporal_tod: bool = False,
        disable_base_prediction: bool = False,
        disable_temporal_residual: bool = False,
    ) -> None:
        super().__init__()
        self.share_base_blocks = {
            x.strip() for x in str(share_base_blocks or "base_series,base_encoder,base_prediction,decoder").split(",") if x.strip()
        }
        self.core = CompactForecastingModel(
            seq_len=seq_len,
            horizon=horizon,
            tod_size=tod_size,
            num_layers=num_layers,
            model_dim=model_dim,
            prompt_dim=prompt_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            disable_series_decomposition=disable_series_decomposition,
            disable_temporal_tod=disable_temporal_tod,
            disable_base_prediction=disable_base_prediction,
            disable_temporal_residual=disable_temporal_residual,
        )
        if use_private_graph_residual:
            if graph_adj is None:
                raise ValueError("use_private_graph_residual=True requires graph_adj")
            if int(graph_adj.shape[0]) != int(num_nodes):
                raise ValueError(
                    f"graph_adj nodes {int(graph_adj.shape[0])} != num_nodes {int(num_nodes)}"
                )
            self.private_graph_residual = SparseGraphResidualAdapter(
                seq_len=seq_len,
                horizon=horizon,
                graph_adj=graph_adj,
                hidden_dim=private_graph_hidden_dim,
                num_layers=private_graph_num_layers,
                gate_init=private_graph_gate_init,
                gate_kind=private_graph_gate_kind,
                residual_scale=private_graph_residual_scale,
                start_step=private_graph_start_step,
                use_tod=private_graph_use_tod,
                init_scale=private_graph_init_scale,
                topk=private_graph_topk,
            )
        else:
            self.private_graph_residual = None
        self.register_buffer("speed_mean", torch.tensor(float(speed_mean)))
        self.register_buffer("speed_std", torch.tensor(float(speed_std)))

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Return denormalized traffic forecasts.

        Input shape is [B, N, T, C], with channels speed, time-of-day, and
        optional day-of-week. Output shape is [B, N, horizon].
        """
        _bsz, _num_nodes, _seq_len, channels = x_input.shape
        if channels < 2:
            raise ValueError(f"expected at least speed and time-of-day channels, got {channels}")
        speed = x_input[..., 0]
        tod = x_input[..., 1]
        dow = x_input[..., 2] if channels >= 3 else torch.zeros_like(tod)
        history = torch.stack([speed, tod, dow], dim=-1).permute(0, 2, 1, 3).contiguous()
        out_norm = self.core(history).squeeze(-1).transpose(1, 2)
        if self.private_graph_residual is not None:
            out_norm = out_norm + self.private_graph_residual(speed, tod)
        return out_norm * self.speed_std + self.speed_mean

    @torch.no_grad()
    def get_shared_state(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if self._is_shared_key(k)
        }

    @torch.no_grad()
    def set_shared_state(self, shared: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for k, v in shared.items():
            if k not in own:
                raise KeyError(f"shared key {k!r} not in local state_dict")
            own[k].data.copy_(v.data)


    def shared_param_names(self) -> list[str]:
        return [k for k in self.state_dict().keys() if self._is_shared_key(k)]

    def private_param_names(self) -> list[str]:
        return [k for k, _ in self.named_parameters() if not self._is_shared_key(k)]

    def efficiency_audit(self) -> dict:
        graph = self.private_graph_residual.efficiency_audit() if self.private_graph_residual else None
        return {"private_graph_residual": graph}

    def _is_shared_key(self, name: str) -> bool:
        stripped = name.replace("core.", "", 1)
        if stripped.startswith("base.time_series_"):
            return "base_series" in self.share_base_blocks
        if stripped.startswith("base.encoder."):
            return "base_encoder" in self.share_base_blocks
        if stripped.startswith("base.prediction_head."):
            return "base_prediction" in self.share_base_blocks
        if stripped.startswith("decoder."):
            return "decoder" in self.share_base_blocks
        return False
