"""Source-only node/day optimization with projected joint clipping and noise."""
from __future__ import annotations
import numpy as np
import torch
from . import protocol as p
from .record_gradients import per_record_temporal_gradients, poisson_inclusion_indices, public_dropout_seed


class SourceTrainer:
    def __init__(self, model, city, stats, *, seed, city_index, expected_batch=128, clip=0.25, sigma=0.8261):
        self.model, self.city, self.stats = model, city, stats
        self.seed, self.city_index = seed, city_index
        self.batch, self.clip, self.sigma = expected_batch, clip, sigma
        if not 0 < expected_batch <= city.population or clip <= 0 or not np.isfinite(sigma) or sigma < 0:
            raise ValueError('Invalid private batch size, clipping bound, or noise multiplier.')
        self.named = {f'core.{k}': v for k, v in model.core.named_parameters()}
        self.shapes = {k: tuple(v.shape) for k, v in self.named.items()}
        self.optimizer = torch.optim.Adam(list(self.named.values()), lr=0.001)
        self.sampler, self.noise = np.random.default_rng(), np.random.default_rng()
        self.step_count = 0

    def train_round(self, shared, bases, round_id, steps):
        device = next(self.model.parameters()).device
        self.model.set_shared_state(shared)
        self.model.train()
        reference = {k: v.to(device) for k, v in shared.items()}
        accumulated = {k: torch.zeros(spec.coefficient_count, device=device) for k, spec in bases.items()}
        for local_step in range(steps):
            ids = poisson_inclusion_indices(self.city.population, self.batch / self.city.population, self.sampler)
            if len(ids):
                records = [self.city.record(atom, self.stats) for atom in ids]
                x = torch.stack([row[0] for row in records]).to(device)
                y = torch.stack([row[1] for row in records]).to(device)
                # Public dropout does not consume the target's training RNG stream.
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                with torch.random.fork_rng(devices=cuda_devices):
                    raw_core = per_record_temporal_gradients(self.model.core, x, y, *self.stats,
                        public_dropout_seed(self.city_index, self.step_count, self.seed))
                raw = {f'core.{k}': v for k, v in raw_core.items()}
            else:
                raw = {k: torch.empty((0,)+shape, device=device) for k, shape in self.shapes.items()}
            clipped = p.map_record_gradients(raw, self.shapes, bases, clip_C=self.clip)
            noise = {k: torch.from_numpy(self.noise.normal(0., self.clip*self.sigma, tuple(v.shape)).astype(np.float32)).to(device)
                     for k, v in clipped.items()}
            gradients = p.processed_full_gradients(clipped, noise, self.shapes, bases, divisor=self.batch)
            self.step_count += 1
            if round_id == 0:
                p.native_adam_step(self.named, self.optimizer, gradients, self.step_count)
            else:
                accumulated = p.projected_displacement_step(self.named, self.optimizer, gradients,
                    reference, bases, accumulated, self.step_count)
        if round_id == 0:
            return {k: p.project_tensor(self.named[k].detach() - reference[k], bases[k]).cpu() for k in bases}
        return {k: v.detach().cpu() for k, v in accumulated.items()}
