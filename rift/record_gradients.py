"""Per-node/day temporal gradients, before projection and joint clipping."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def per_record_temporal_gradients(core: torch.nn.Module, histories: torch.Tensor, raw_labels: torch.Tensor,
                                  released_mean: float, released_std: float,
                                  public_dropout_seed_value: int) -> dict[str, torch.Tensor]:
    """Return complete gradients per record; caller clips each record across all parameters."""
    if released_std <= 0:
        raise ValueError("released P012 standard deviation must be positive")
    from torch.func import functional_call, grad, vmap

    parameters = dict(core.named_parameters())

    def one_record_loss(params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        normalized_prediction = functional_call(core, params, (x,)).squeeze(-1).transpose(1, 2)
        if normalized_prediction.shape != y.shape:
            raise ValueError("native temporal-core prediction and raw node/day labels differ in shape")
        raw_prediction = normalized_prediction * float(released_std) + float(released_mean)
        return F.mse_loss(raw_prediction, y)

    torch.manual_seed(int(public_dropout_seed_value))
    return vmap(grad(one_record_loss), in_dims=(None, 0, 0), randomness="different")(
        parameters, histories, raw_labels
    )


def poisson_inclusion_indices(population: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """Equivalent binomial-count/uniform-subset implementation of Poisson inclusion."""
    if population < 1 or not (0.0 < q <= 1.0):
        raise ValueError("public population and fixed inclusion probability must be positive")
    count = int(rng.binomial(int(population), float(q)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(rng.choice(int(population), size=count, replace=False), dtype=np.int64)


def public_dropout_seed(city_index: int, logical_step: int, model_seed: int) -> int:
    """Public model-randomness stream, disjoint from secret sampling/noise streams."""
    return 13012 + 100000 * int(city_index) + int(logical_step) + int(model_seed)

