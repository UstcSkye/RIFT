"""Memory-mapped city data and interval-contained forecasting windows."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

SPEED_CITIES = ('metr-la', 'pems-bay', 'chengdu', 'shenzhen')
FLOW_CITIES = ('sd', 'oc', 'la', 'gba')
CITIES = SPEED_CITIES + FLOW_CITIES


def group(city):
    return SPEED_CITIES if city in SPEED_CITIES else FLOW_CITIES


def steps_per_day(city):
    return 144 if city in ('chengdu', 'shenzhen') else 288


def map_domain(values, bound):
    x = np.asarray(values, dtype=np.float64)
    return np.clip(np.nan_to_num(x, nan=0., posinf=bound, neginf=0.), 0., bound).astype(np.float32)


class CityData:
    def __init__(self, root, city, *, synthetic=False):
        self.city = city
        self.flow = city in FLOW_CITIES
        self.day_steps = steps_per_day(city)
        self.bound = 4096.0 if self.flow else 200.0
        folder = Path(root) / city
        if city == 'oc' and not folder.exists():
            folder = Path(root) / 'orange'  # Original LargeST archive directory.
        self.folder = folder
        self.array = np.load(folder / 'dataset.npy', mmap_mode='r', allow_pickle=False)
        if self.array.ndim != 3 or self.array.shape[2] < (1 if self.flow else 2):
            raise ValueError(f'{city}: expected dataset.npy [T,N,C], value plus raw ToD for speed.')
        self.steps, self.nodes = self.array.shape[:2]
        # LargeST uses the source prefix that precedes the common development interval.
        self.fit_end = 73584 if self.flow and not synthetic else math.floor(self.steps * .8)
        if self.fit_end > self.steps:
            raise ValueError(f'{city}: LargeST requires the prepared 2019 sequence; found only {self.steps} steps.')
        self.days = self.fit_end // self.day_steps
        self.population = self.nodes * self.days
        if self.days < 1:
            raise ValueError('Source interval must contain at least one complete day.')
        self.synthetic = synthetic

    def values(self, begin, end, node=None, *, source=False):
        values = np.asarray(self.array[begin:end, :, 0] if node is None else self.array[begin:end, node, 0], dtype=np.float32)
        if source:
            return map_domain(values, self.bound)
        if not np.isfinite(values).all():
            raise ValueError(f'{self.city}: target values must be finite; use the prepared dataset.')
        return np.array(values, copy=True)

    def tod(self, begin, end, node=None):
        if self.flow:
            values = (np.arange(begin, end) % self.day_steps).astype(np.float32) / self.day_steps
            return values if node is not None else np.broadcast_to(values[:, None], (end-begin, self.nodes)).copy()
        x = np.asarray(self.array[begin:end, :, 1] if node is None else self.array[begin:end, node, 1], dtype=np.float32)
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x >= 1):
            raise ValueError(f'{self.city}: time-of-day must be the unnormalized phase in [0,1).')
        return np.array(x, copy=True)

    def graph(self):
        a = np.load(self.folder / 'matrix.npy', allow_pickle=False).astype(np.float32)
        if a.shape != (self.nodes, self.nodes) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('matrix.npy must be a finite, nonnegative [N,N] adjacency.')
        a = a + np.eye(self.nodes, dtype=np.float32)
        d = np.maximum(a.sum(axis=1), 1e-5)
        inv = np.reciprocal(np.sqrt(d)).astype(np.float32)
        return torch.from_numpy(inv[:, None] * a * inv[None, :])

    def support_statistics(self, target_days):
        values = self.values(0, target_days * self.day_steps)
        mean = np.float32(values.mean(dtype=np.float64))
        std = np.float32(values.std(dtype=np.float64))
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError('Target support must have a positive finite standard deviation.')
        return [float(mean), float(std)]

    def interval(self, split):
        if self.flow and not self.synthetic:
            interval = (76032, 84096) if split == 'dev' else (84096, 105120)
        else:
            test_start = math.floor(.8 * self.steps)
            interval = (max(0, test_start // self.day_steps * self.day_steps - 28*self.day_steps),
                        test_start // self.day_steps * self.day_steps) if split == 'dev' else (test_start, self.steps)
        if interval[1] > self.steps or interval[1] - interval[0] < 18:
            raise ValueError(f'{self.city}: dataset too short for {split} interval {interval}.')
        return interval

    def record(self, atom, stats):
        node, day = divmod(int(atom), self.days)
        begin = day * self.day_steps
        raw = self.values(begin, begin+self.day_steps, node, source=True)
        tod = self.tod(begin, begin+self.day_steps, node)
        starts = np.linspace(0, self.day_steps-18, 16, dtype=np.int64)
        histories, labels = [], []
        for start in starts:
            norm = (raw[start:start+12] - stats[0]) / stats[1]
            histories.append(np.stack([norm, tod[start:start+12], np.zeros(12, np.float32)], -1))
            labels.append(raw[start+12:start+18])
        return torch.from_numpy(np.stack(histories)).unsqueeze(2), torch.from_numpy(np.stack(labels)).unsqueeze(1)


class Windows(Dataset):
    def __init__(self, city, interval, stats):
        self.city, self.begin, self.end, self.stats = city, *interval, stats
        if self.begin < 0 or self.end > city.steps or len(self) < 1:
            raise ValueError('Invalid window interval.')

    def __len__(self):
        return self.end - self.begin - 17

    def __getitem__(self, index):
        start = self.begin + index
        raw = self.city.values(start, start+18)
        tod = self.city.tod(start, start+12)
        x = np.stack([(raw[:12] - self.stats[0]) / self.stats[1], tod], -1).transpose(1, 0, 2)
        return torch.from_numpy(x.copy()), torch.from_numpy(raw[12:].T.copy())


def source_statistics(city, cache_dir):
    """Release once and cache two noisy bounded sensor/day statistics.

    Each query uses replacement zCDP rho=.05. Training privacy accounting must
    also include this preprocessing release under the chosen adjacency.
    """
    path = Path(cache_dir) / (city.city + '.json')
    metadata = dict(city=city.city, shape=list(city.array.shape), complete_days=city.days,
                    domain=[0, city.bound], rho=0.1, adjacency='sensor-day replacement')
    if path.exists():
        cached = json.loads(path.read_text())
        if cached['metadata'] != metadata:
            raise ValueError(f'{path}: statistics belong to a different public data geometry.')
        stats = cached['statistics']
        if not np.isfinite(stats).all() or stats[1] <= 0:
            raise ValueError('Invalid cached source statistics.')
        return stats
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()  # Separate secret randomness, never --seed.
    count = city.population * city.day_steps
    def chunks():
        stop = city.days * city.day_steps
        for begin in range(0, stop, 1024):
            yield city.values(begin, min(stop, begin+1024), source=True).astype(np.float64)
    total = sum(float(chunk.sum(dtype=np.float64)) for chunk in chunks())
    mu = np.float32(np.clip(total / count + rng.normal(0, (city.bound / city.population) / math.sqrt(.1)), 0, city.bound))
    second = sum(float(np.square(chunk-float(mu)).sum(dtype=np.float64)) for chunk in chunks()) / count
    sensitivity = max(float(mu)**2, (city.bound-float(mu))**2) / city.population
    std = np.float32(math.sqrt(max(1., second + rng.normal(0, sensitivity/math.sqrt(.1)))))
    stats = [float(mu), float(std)]
    # Do not overwrite an existing release, including one from another process.
    with path.open('x', encoding='utf-8') as f:
        json.dump(dict(metadata=metadata, statistics=stats), f, indent=2)
    return stats
