"""Target-conditioned innovation weighting with softmax and EMA."""
import math
import numpy as np
import torch


class TargetConditionedAggregator:
    def __init__(self, source_count=3, target_weight=0.9, temperature=1.0, smoothing=0.25):
        if source_count < 1 or temperature <= 0 or not 0 <= target_weight <= 1 or not 0 <= smoothing < 1:
            raise ValueError('Invalid aggregation parameters.')
        self.weights = np.full(source_count, 1.0 / source_count, dtype=np.float64)
        self.target_weight, self.temperature, self.smoothing = target_weight, temperature, smoothing

    @staticmethod
    def cosine(left, right):
        a = torch.cat([x.cpu().reshape(-1).float() for x in left.values()])
        b = torch.cat([right[k].cpu().reshape(-1).float() for k in left])
        denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
        return float((a*b).sum() / (a.norm()*b.norm())) if denominator > 1e-12 else 0.0

    def __call__(self, reference, target_update, source_updates, innovations):
        scores = np.array([self.cosine(e, target_update) for e in innovations], dtype=np.float64)
        soft = np.array([math.exp((float(v)-float(scores.max()))/self.temperature) for v in scores])
        soft /= soft.sum()
        self.weights = self.smoothing * self.weights + (1 - self.smoothing) * soft
        self.weights /= self.weights.sum()
        result = {}
        for name in reference:
            mixture = target_update[name].cpu().float() * self.target_weight
            for weight, source in zip(self.weights, source_updates):
                mixture = mixture + source[name].cpu().float() * ((1 - self.target_weight) * float(weight))
            result[name] = reference[name].cpu().float() + mixture
        return result
